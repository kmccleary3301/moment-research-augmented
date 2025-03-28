import os
import warnings
from copy import deepcopy

import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch import optim
from wandb import AlertLevel
import random
from datetime import datetime

from moment.common import PATHS
from moment.data.dataloader import get_timeseries_dataloader
from moment.models.anomaly_nearest_neighbors import AnomalyNearestNeighbors
from moment.models.anomaly_transformer import AnomalyTransformer
from moment.models.base import BaseModel
from moment.models.dghl import DGHL
from moment.models.dlinear import DLinear
from moment.models.gpt4ts import GPT4TS
from moment.models.moment import MOMENT
from moment.models.nbeats import NBEATS
from moment.models.nhits import NHITS
from moment.models.timesnet import TimesNet
from moment.utils.forecasting_metrics import sMAPELoss
from moment.utils.optims import LinearWarmupCosineLRScheduler
from moment.utils.utils import MetricsStore
from moment.models.tinytimemixer.adapter import TinyTimeMixerAdapter
from moment.models.tinytimemixer.attempt_02 import AugmentedTTM
from moment.models.titans.attempt_01 import MAC_Moment

warnings.filterwarnings("ignore")


class WrappedLogger():
    def __init__(
        self, 
        task_superclass,
        use_wandb: bool = True,
        output_dir: str = None,
        config: dict = None,
        project_name: str = "Time-series Foundation Model",
        name: str = None,
        notes: str = None,
    ):
        self.name = name
        self.notes = task_superclass.args.notes if notes is None else notes
        self.use_wandb = use_wandb
        self.dir = output_dir
        self.project = project_name
        # self.config = task_superclass.args
        
        if use_wandb:
            print("Initializing wandb logger")
            self.logger = wandb.init(
                project=project_name,
                dir=output_dir,
                config=task_superclass.args,
                name=self.name,
                notes=task_superclass.args.notes if notes is None else notes,
                mode="disabled" if task_superclass.args.debug else "run",
            )
        else:
            print("Using dummy logger")

    def log(self, data):
        if self.use_wandb:
            self.logger.log(data)
        else:
            print(data)
            
    def alert(self, title, text, level):
        if self.use_wandb:
            self.logger.alert(title, text, level)
        else:
            print(f"{title} - {text}")
    
    def finish(self):
        if self.use_wandb:
            self.logger.finish()
        else:
            print("Logger finished")


class Tasks(nn.Module):
    def __init__(self, args, **kwargs):
        super(Tasks, self).__init__()
        self.args = args
        self._dataloader = {}

        self.seed = getattr(args, "seed", random.randint(0, 1000000))
        
        print(f"Setting seed to {self.seed}")
        
        self._set_seed()
        
        self._build_model()
        self._acquire_device()

        # Setup data loaders
        
        
        # Need to reset seed after building model
        # This is because the model is built using the seed.
        # So different models will result in different data
        # unless we reset the seed again.
        self._set_seed() 
        self.train_dataloader = self._get_dataloader(data_split="train")
        self.test_dataloader = self._get_dataloader(data_split="test")
        self.val_dataloader = self._get_dataloader(data_split="val")


    def _set_seed(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)

    def _build_model(self):
        self.model_name = self.args.model_name
        
        if self.args.model_name == "MOMENT":
            self.model = MOMENT(configs=self.args)
        elif self.args.model_name == "DLinear":
            self.model = DLinear(configs=self.args)
        elif self.args.model_name == "DGHL":
            self.model = DGHL(configs=self.args)
        elif self.args.model_name == "AnomalyTransformer":
            self.model = AnomalyTransformer(configs=self.args)
        elif self.args.model_name == "AnomalyNearestNeighbors":
            self.model = AnomalyNearestNeighbors(configs=self.args)
        elif self.args.model_name == "N-BEATS":
            self.model = NBEATS(configs=self.args)
        elif self.args.model_name == "N-HITS":
            self.model = NHITS(configs=self.args)
        elif self.args.model_name == "GPT4TS":
            self.model = GPT4TS(configs=self.args)
        elif self.args.model_name == "TimesNet":
            self.model = TimesNet(configs=self.args)
        elif self.args.model_name == "TinyTimeMixer":
            self.model = TinyTimeMixerAdapter(configs=self.args)
        elif self.args.model_name == "AugmentedTTM":
            self.model = AugmentedTTM(config=self.args)
        elif self.args.model_name == "MACMoment":
            self.model = MAC_Moment(configs=self.args)
        else:
            raise NotImplementedError(f"Model {self.args.model_name} not implemented")
        
        
        # Get a nice model identifier
        try:
            self.total_model_params = sum(p.numel() for p in self.model.parameters())
            total_params, prefix = sum(p.numel() for p in self.model.parameters()), ""
        
            for value in ["K", "M", "B"]:
                if total_params < 1000:
                    break
                total_params /= 1000
                prefix = value
            
            self.model_label = f"{self.model_name} {total_params:.1f}{prefix}"
        except:
            self.model_params = 0
            self.model_label = self.model_name
        
        return self.model

    def _acquire_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(self.args.device))
        else:
            self.device = torch.device("cpu")
        return self.device

    def _reset_dataloader(self):
        self._dataloader = {}

    def _get_dataloader(self, data_split: str = "train"):
        # Load Datasets
        if self._dataloader.get(data_split) is not None:
            return self._dataloader.get(data_split)
        else:
            data_loader_args = deepcopy(self.args)
            data_loader_args.data_split = data_split
            if self.args.task_name == "pre-training":
                data_loader_args.dataset_names = "all"
            data_loader_args.batch_size = (
                self.args.train_batch_size
                if data_split == "train"
                else self.args.val_batch_size
            )
            print(f"Loading {data_split} split of the dataset")

            self._dataloader[data_split] = get_timeseries_dataloader(
                args=data_loader_args
            )
            return self._dataloader.get(data_split)

    def _select_optimizer(self):
        if self.args.optimizer_name == "AdamW":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.args.init_lr,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer_name == "Adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.init_lr,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer_name == "SGD":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.init_lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"Optimizer {self.args.optimizer_name} not implemented"
            )
        return optimizer

    def _init_lr_scheduler(self, type: str = "linearwarmupcosinelr"):
        decay_rate = self.args.lr_decay_rate
        warmup_start_lr = self.args.warmup_lr
        warmup_steps = self.args.warmup_steps

        if type == "linearwarmupcosinelr":
            self.lr_scheduler = LinearWarmupCosineLRScheduler(
                optimizer=self.optimizer,
                max_epoch=self.args.max_epoch,
                min_lr=self.args.min_lr,
                init_lr=self.args.init_lr,
                decay_rate=decay_rate,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
            )
        elif type == "onecyclelr":
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.args.init_lr,
                epochs=self.args.max_epoch,
                steps_per_epoch=len(self.train_dataloader),
                pct_start=self.args.pct_start,
            )
        elif type == "none":
            self.lr_scheduler = None

    def _select_criterion(
        self, loss_type: str = "mse", reduction: str = "none", **kwargs
    ):
        if loss_type == "mse":
            criterion = nn.MSELoss(reduction=reduction)
        elif loss_type == "mae":
            criterion = nn.L1Loss(reduction=reduction)
        elif loss_type == "huber":
            criterion = nn.HuberLoss(reduction=reduction, delta=kwargs["delta"])
        elif loss_type == "smape":
            criterion = sMAPELoss(reduction=reduction)
        return criterion

    def save_results(self, results_df: pd.DataFrame, path: str, opt_steps: int):
        results_df.to_csv(
            os.path.join(path, f"results_{self.args.task_name}_{opt_steps}.csv")
        )

    def save_model(
        self,
        model: nn.Module,
        path: str,
        opt_steps: int,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
    ):
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        }

        if opt_steps is None:
            with open(os.path.join(path, f"{self.args.model_name}.pth"), "wb") as f:
                torch.save(checkpoint, f)
        else:
            with open(
                os.path.join(
                    path, f"{self.args.model_name}_checkpoint_{opt_steps}.pth"
                ),
                "wb",
            ) as f:
                torch.save(checkpoint, f)

    def save_model_and_alert(self, opt_steps):
        self.save_model(
            self.model, self.checkpoint_path, opt_steps, self.optimizer, self.scaler
        )

    def load_pretrained_model(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage.cuda(self.device)
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def load_pretrained_moment(
        self, pretraining_task_name: str = "pre-training", do_not_copy_head: bool = True
    ):
        pretraining_args = deepcopy(self.args)
        pretraining_args.task_name = pretraining_task_name

        checkpoint = BaseModel.load_pretrained_weights(
            run_name=pretraining_args.pretraining_run_name,
            opt_steps=pretraining_args.pretraining_opt_steps,
        )

        pretrained_model = MOMENT(configs=pretraining_args)
        pretrained_model.load_state_dict(checkpoint["model_state_dict"])

        # Copy pre-trained parameters to fine-tuned model
        for (name_p, param_p), (name_f, param_f) in zip(
            pretrained_model.named_parameters(), self.model.named_parameters()
        ):
            if (name_p == name_f) and (param_p.shape == param_f.shape):
                if do_not_copy_head and name_p.startswith("head"):
                    continue
                else:
                    param_f.data = param_p.data

        self.freeze_model_parameters()  # Freeze model parameters based on fine-tuning mode

        return True

    def freeze_model_parameters(self):
        if self.args.finetuning_mode == "linear-probing":
            for name, param in self.model.named_parameters():
                if not name.startswith("head"):
                    param.requires_grad = False
        elif self.args.finetuning_mode == "end-to-end":
            pass
        else:
            raise NotImplementedError(
                f"Finetuning mode {self.args.finetuning_mode} not implemented"
            )

        print("====== Frozen parameter status ======")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print("Not frozen:", name)
            else:
                print("Frozen:", name)
        print("=====================================")

    def _create_results_dir(self, experiment_name="supervised_forecasting"):
        if experiment_name == "supervised_forecasting":
            results_path = os.path.join(
                PATHS.RESULTS_DIR,
                experiment_name,
                self.args.model_name,
                self.dataset_name_,
                self.args.finetuning_mode,
                str(self.args.forecast_horizon),
            )
        elif experiment_name == "supervised_anomaly_detection":
            results_path = os.path.join(
                PATHS.RESULTS_DIR,
                experiment_name,
                self.args.model_name,
                self.args.finetuning_mode,
            )
        elif experiment_name == "supervised_imputation":
            results_path = os.path.join(
                PATHS.RESULTS_DIR,
                experiment_name,
                self.args.model_name,
                self.dataset_name_,
                self.args.finetuning_mode,
            )

        os.makedirs(results_path, exist_ok=True)
        return results_path

    def setup_logger(
        self, 
        notes: str = None,
        project_name: str = "Time-series Foundation Model",
        output_dir: str | None = None
    ):
        
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if output_dir is None:
            dir_name = f"{current_datetime}_{self.model_label}"
            output_dir = os.path.join("runs", dir_name)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        use_wandb = getattr(self.args, "use_wandb", True)
        print(f"Using wandb: {use_wandb}")
        if use_wandb: # Maybe it doesn't work if it initializes inside the wrapped logger
            self.logger = wandb.init(
                project="Time-series Foundation Model",
                dir=output_dir,
                config=self.args,
                name=self.args.run_name if hasattr(self.args, "run_name") else self.model_label,
                notes=self.args.notes if notes is None else notes,
                mode="disabled" if self.args.debug else "run",
            )
        
        self.wrapped_logger = WrappedLogger(
            task_superclass=self,
            project_name=project_name,
            output_dir=output_dir,
            config=self.args,
            name=self.args.run_name if hasattr(self.args, "run_name") else self.model_label,
            notes=self.args.notes if notes is None else notes,
            # mode="disabled" if self.args.debug else "run",
            use_wandb=False,
        )
        if self.args.debug:
            print(f"Run name: {self.logger.name}\n")
        
        print("Saving model to output directory:", output_dir)    
        
        with open(os.path.join(output_dir, "model_print.txt"), "w") as f:
            f.write(str(self.model))
        
        return self.logger

    def end_logger(self):
        self.logger.finish()

    def evaluate_model(self):
        return MetricsStore(
            train_loss=self.validation(self.train_dataloader),
            test_loss=self.validation(self.test_dataloader),
            val_loss=self.validation(self.val_dataloader),
        )

    def evaluate_and_log(self):
        eval_metrics = self.evaluate_model()
        self.logger.log(
            {
                "train_loss": eval_metrics.train_loss,
                "validation_loss": eval_metrics.val_loss,
                "test_loss": eval_metrics.test_loss,
            }
        )
        return eval_metrics

    def debug_model_outputs(self, loss, outputs, batch_x, **kwargs):
        # Debugging code
        if (
            torch.any(torch.isnan(loss))
            or torch.any(torch.isinf(loss))
            or (loss < 1e-3)
        ):
            self.logger.alert(
                title="Loss is NaN or Inf or too small",
                text=f"Loss is {loss.item()}.",
                level=AlertLevel.INFO,
            )
            breakpoint()

        # Check model outputs
        if outputs.illegal_output:
            self.logger.alert(
                title="Model weights are NaN or Inf",
                text=f"Model weights are NaN or Inf.",
                level=AlertLevel.INFO,
            )
            breakpoint()

        # Check model gradients
        illegal_encoder_grads = (
            torch.stack(
                [torch.isfinite(p).any() for p in self.model.encoder.parameters()]
            )
            .any()
            .item()
        )
        illegal_head_grads = (
            torch.stack([torch.isfinite(p).any() for p in self.model.head.parameters()])
            .any()
            .item()
        )
        illegal_patch_embedding_grads = (
            torch.stack(
                [
                    torch.isfinite(p).any()
                    for p in self.model.patch_embedding.parameters()
                ]
            )
            .any()
            .item()
        )

        illegal_grads = (
            illegal_encoder_grads or illegal_head_grads or illegal_patch_embedding_grads
        )
        
        zipped_illegal_grads = zip(
            [illegal_encoder_grads, illegal_head_grads, illegal_patch_embedding_grads], 
            ["encoder", "head", "patch_embedding"]
        )

        # if illegal_grads:
        #     # self.logger.alert(title="Model gradients are NaN or Inf",
        #     #                     text=f"Model gradients are NaN or Inf.",
        #     #                     level=AlertLevel.INFO)
        #     # breakpoint()
            
        #     grad_flags = [(grad_label, bool(grad_bool)) for grad_bool, grad_label in zipped_illegal_grads]
            
        #     print(f"Model gradients are NaN or Inf for {str(grad_flags)}.")

        return
