import os
import subprocess
import warnings
import json
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm
from wandb import AlertLevel

from moment.common import PATHS
from moment.models.moment import MOMENT
from moment.utils.utils import MetricsStore, dtype_map, make_dir_if_not_exists
from moment.utils.hashing import sha256_of_tensor, sha256_of_string, sha256_of_array
from moment.data.base import TimeseriesData

from .base import Tasks, AugmentedTTM

warnings.filterwarnings("ignore")


class Pretraining(Tasks):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.args = args
        self.get_dataset_signatures()
    
    def validation(self, data_loader, return_preds: bool = False):
        trues, preds, masks, losses = [], [], [], []

        self.model.eval()
        with torch.no_grad():
            for batch_x in tqdm(data_loader, total=len(data_loader)):
                timeseries = batch_x.timeseries.float().to(self.device)
                input_mask = batch_x.input_mask.long().to(self.device)

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype_map(self.args.torch_dtype),
                    enabled=self.args.use_amp,
                ):
                    outputs = self.model(
                        x_enc=timeseries, input_mask=input_mask, mask=None
                    )

                recon_loss = self.criterion(outputs.reconstruction, timeseries)
                observed_mask = input_mask * (1 - outputs.pretrain_mask)
                n_channels = outputs.reconstruction.shape[1]
                observed_mask = observed_mask.unsqueeze(1).repeat((1, n_channels, 1))
                masked_loss = observed_mask * recon_loss
                loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)

                losses.append(loss.item())

                if return_preds:
                    trues.append(timeseries.detach().cpu().numpy())
                    preds.append(outputs.reconstruction.detach().cpu().numpy())
                    masks.append(outputs.pretrain_mask.detach().cpu().numpy())

        losses = np.array(losses)
        average_loss = np.average(losses)
        self.model.train()

        if return_preds:
            trues = np.concatenate(trues, axis=0)
            preds = np.concatenate(preds, axis=0)
            masks = np.concatenate(masks, axis=0)
            return average_loss, losses, (trues, preds, masks)
        else:
            return average_loss
        
    def validation_on_subset(self, subset: list, return_preds: bool = False):
        trues, preds, masks, losses = [], [], [], []

        self.model.eval()
        with torch.no_grad():
            for batch_x in subset:
                timeseries = batch_x.timeseries.float().to(self.device)
                input_mask = batch_x.input_mask.long().to(self.device)

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype_map(self.args.torch_dtype),
                    enabled=self.args.use_amp,
                ):
                    outputs = self.model(
                        x_enc=timeseries, input_mask=input_mask, mask=None
                    )

                recon_loss = self.criterion(outputs.reconstruction, timeseries)
                observed_mask = input_mask * (1 - outputs.pretrain_mask)
                n_channels = outputs.reconstruction.shape[1]
                observed_mask = observed_mask.unsqueeze(1).repeat((1, n_channels, 1))
                masked_loss = observed_mask * recon_loss
                loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)

                losses.append(loss.item())

                if return_preds:
                    trues.append(timeseries.detach().cpu().numpy())
                    preds.append(outputs.reconstruction.detach().cpu().numpy())
                    masks.append(outputs.pretrain_mask.detach().cpu().numpy())

        losses = np.array(losses)
        average_loss = np.average(losses)
        self.model.train()

        if return_preds:
            trues = np.concatenate(trues, axis=0)
            preds = np.concatenate(preds, axis=0)
            masks = np.concatenate(masks, axis=0)
            return average_loss, losses, (trues, preds, masks)
        else:
            return average_loss

    def get_dataset_signatures(self):
        # We'll hash the data loaders to validate complete reproducibility
        # Across model tests, ensuring that all batches are processed in the same order
        # and that everything is exactly the same
        # Essentially, we are guaranteeing that circumstances are completely identical
        # across model tests
        data_loaders_sample_sets = {k: set() for k in ["train", "val", "test"]}
        hash_data_loaders_thorough = {k: "a" for k in ["train", "val", "test", "full"]}
        for (loader, key) in zip(
            [self.train_dataloader, self.val_dataloader, self.test_dataloader], 
            ["train", "val", "test"]
        ):
            for batch in tqdm(loader, desc=f"Hashing {key} data"):
                batch : TimeseriesData = batch
                
                try:
                    all_hashes = [
                        sha256_of_array(e) if not e is None else sha256_of_string("None") for e in [
                            batch.timeseries,
                            batch.forecast,
                            batch.labels,
                            batch.input_mask,
                            batch.metadata,
                            batch.name
                        ]
                    ]
                    
                    sample_hash = sha256_of_array(batch.timeseries)
                    hash_data_loaders_thorough[key] = sha256_of_string("".join(all_hashes))
                    data_loaders_sample_sets[key].add(sample_hash)
                except Exception as e:
                    print(f"Error hashing time_series: {e}")
                    raise e
                
                # hash_data_loaders_thorough[key] += sample_hash
                # hash_data_loaders_thorough[key] += sha256_of_tensor(batch["mask"])
                # hash_data_loaders_thorough[key] += sha256_of_tensor(batch["idx"])
                
                # hash_data_loaders_thorough[key] = sha256_of_string(hash_data_loaders_thorough[key])
            
            hash_data_loaders_thorough["full"] += hash_data_loaders_thorough[key]
        hash_data_loaders_thorough["full"] = sha256_of_string(hash_data_loaders_thorough["full"])
        
        
        # Check that intersections are empty
        for key in data_loaders_sample_sets:
            for other_key in data_loaders_sample_sets:
                if key == other_key:
                    continue
                
                intersection = data_loaders_sample_sets[key].intersection(data_loaders_sample_sets[other_key])
                assert len(intersection) == 0, \
                    f"Intersections are not empty for {key} and {other_key} ({len(intersection)} samples in common)"
        
        with open(os.path.join(self.args.checkpoint_path, "dataset_signatures.json"), "w") as f:
            json.dump(hash_data_loaders_thorough, f)

        print("="*20, "Dataset signatures", "="*20)
        print(json.dumps(hash_data_loaders_thorough, indent=4))
        print("="*50)
        

    def train(self):
        self.run_name = self.logger.name
        print("Run name:", self.run_name)
        print("Logger:", vars(self.logger))
        
        first_print = True
        
        path = os.path.join(self.args.checkpoint_path, self.run_name)
        make_dir_if_not_exists(path, verbose=True)

        self.optimizer = self._select_optimizer()
        self.criterion = self._select_criterion()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        self._init_lr_scheduler()

        self.model.to(self.device)

        opt_steps = 0
        cur_epoch = 0
        
        val_batches = [val_batch for val_batch in self.val_dataloader]
        
        if isinstance(self.model, AugmentedTTM):
            self.model.config.task_name = "pretrain"
        
        
        while (opt_steps < self.args.max_opt_steps) or (cur_epoch < self.args.max_epoch):
            self.model.train()
            
            last_val_idx, current_pbar_dict = 0, {}
            
            
            print(f"Training on {self.args.max_opt_steps} steps or {self.args.max_epoch} epochs")
            
            pbar = tqdm(
                self.train_dataloader, 
                total=len(self.train_dataloader),
                desc=f"Epoch {cur_epoch+1}/{self.args.max_epoch}"
            )
            for batch_idx, batch_x in enumerate(pbar):
                current_epoch_progress = batch_idx / len(self.train_dataloader)
                
                self.optimizer.zero_grad(set_to_none=True)
                timeseries = batch_x.timeseries.float().to(self.device)
                input_mask = batch_x.input_mask.long().to(self.device)

                if not self.args.set_input_mask:
                    input_mask = torch.ones_like(input_mask)

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype_map(self.args.torch_dtype),
                    enabled=self.args.use_amp,
                ):
                    if first_print:
                        print("Passing inputs to model")
                        print(f"Timeseries: {type(timeseries)} {timeseries.shape}")
                        print(f"Input mask: {type(input_mask)} {input_mask.shape}")
                        
                    outputs = self.model(
                        x_enc=timeseries, input_mask=input_mask, mask=None
                    )

                    if first_print:
                        print(f"Outputs: {type(outputs)}")
                        
                
                if first_print:
                    print("Attempting to calc loss")
                    print(f"Outputs: {type(outputs.reconstruction)} {outputs.reconstruction.shape}")
                    print(f"Timeseries: {type(timeseries)} {timeseries.shape}")
                
                recon_loss = self.criterion(outputs.reconstruction, timeseries)
                
                if first_print:
                    print(f"Recon loss: {type(recon_loss)} {recon_loss.shape}")
                
                observed_mask = input_mask * (1 - outputs.pretrain_mask)
                n_channels = outputs.reconstruction.shape[1]
                observed_mask = observed_mask.unsqueeze(1).repeat((1, n_channels, 1))
                masked_loss = observed_mask * recon_loss
                loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)

                train_loss = loss.item()
                current_pbar_dict["train"] = train_loss
                pbar.set_postfix(current_pbar_dict)
                
                self.logger.log(
                    {
                        "step_train_loss": train_loss,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                )

                if self.args.debug and opt_steps >= 1:
                    self.debug_model_outputs(loss, outputs, batch_x)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                
                
                # Pause and run a batch of val loss
                new_val_idx = int(current_epoch_progress * len(self.val_dataloader))
                if new_val_idx > last_val_idx:
                    subset = val_batches[last_val_idx:new_val_idx]
                    val_loss = self.validation_on_subset(subset)
                    
                    self.logger.log({"validation_loss": val_loss})
                    self.model.train()
                    last_val_idx = new_val_idx
                    current_pbar_dict["val"] = val_loss
                    pbar.set_postfix(current_pbar_dict)
                
                opt_steps = opt_steps + 1

                # if opt_steps % self.args.log_interval == 0:
                #     self.evaluate_and_log()

                if opt_steps % self.args.checkpoint_interval == 0:
                    self.logger.alert(
                        title="Saving model",
                        text=f"Saving model after {opt_steps} steps",
                        level=AlertLevel.INFO,
                    )
                    self.save_model(
                        self.model, path, opt_steps, self.optimizer, self.scaler
                    )
                    # self.evaluate_model_external(path, opt_steps, self.device)

                self.lr_scheduler.step(cur_epoch=cur_epoch, cur_step=opt_steps)

                if first_print:
                    first_print = False
                
            cur_epoch = cur_epoch + 1

        return self.model

    def evaluate_model(self):
        return MetricsStore(val_loss=self.validation(self.val_dataloader))

    def evaluate_and_log(self):
        eval_metrics = self.evaluate_model()
        self.logger.log({"validation_loss": eval_metrics.val_loss})
        return eval_metrics

    def evaluate_model_external(self, path: str, opt_steps: int, device) -> None:
        print("starting evaluation")
        eval_device = int(str(device).split(":")[-1]) + 1
        command = [
            "python",
            "scripts/evaluation/evaluation.py",
            f"--checkpoint_path={path}",
            f"--opt_steps={opt_steps}",
            f"--run_name={self.run_name}",
            f"--gpu_id={eval_device}",
        ]
        outfile = open(f"{path}/eval_output.txt", "w")
        errfile = open(f"{path}/eval_error.txt", "w")
        subprocess.Popen(command, stdout=outfile, stderr=errfile)
