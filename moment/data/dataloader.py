import logging

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.utils import shuffle
from torch.utils.data import ConcatDataset, DataLoader

from .anomaly_detection_datasets import (
    AnomalyDetectionDataset,
    get_anomaly_detection_datasets,
)
from .base import TimeseriesData
from .classification_datasets import ClassificationDataset, get_classification_datasets
from .forecasting_datasets import (
    LongForecastingDataset,
    ShortForecastingDataset,
    get_forecasting_datasets,
)
from .folder_walking import format_file_size
import random
from tqdm import tqdm


def _sample_datasets(dataset_name_to_class: dict, datasets_fraction: float = 1.0):
    if datasets_fraction < 1.0:
        end_idx = int(len(dataset_name_to_class) * datasets_fraction)
        shuffled_items = shuffle(list(dataset_name_to_class.items()))
        dataset_name_to_class = dict(shuffled_items[:end_idx])
    return dataset_name_to_class


def get_all_datasets(
    datasets_fraction: float = 1.0, 
    max_files: int | None = None,
    seed: int = None,
):
    if max_files is None:
        subsets_max_files = [None for _ in range(4)]
    else:
        subsets_max_files = [max_files // 4 for _ in range(4)]
        sums_subsets_max_files = sum(subsets_max_files)
        diff_sums = max_files - sums_subsets_max_files
        for i in range(diff_sums):
            subsets_max_files[i] += 1
        
        assert all([e >= 1 for e in subsets_max_files]), \
            "All four subsets must have at least 1 file, meaning max_files must be at least 4"
    
    classification_datasets = get_classification_datasets(collection="UCR", max_files=subsets_max_files[0], seed=seed)
    forecasting_datasets_long = get_forecasting_datasets(collection="autoformer", max_files=subsets_max_files[1], seed=seed)
    
    forecasting_datasets_short = get_forecasting_datasets(collection="monash", max_files=subsets_max_files[2], seed=seed)
    forecasting_datasets_short = forecasting_datasets_short + get_forecasting_datasets(
        collection="fred/preprocessed", max_files=subsets_max_files[2], seed=seed
    )
    
    if not subsets_max_files[2] is None:
        random.seed(seed)
        random.shuffle(forecasting_datasets_short)
        forecasting_datasets_short = forecasting_datasets_short[:subsets_max_files[2]]
    
    anomaly_detection_datasets = get_anomaly_detection_datasets(
        collection="TSB-UAD-Public", max_files=subsets_max_files[3], seed=seed
    )
    
    all_datasets = classification_datasets + forecasting_datasets_long + \
        forecasting_datasets_short + anomaly_detection_datasets
        
    all_file_entries = sorted(all_datasets, key=lambda x: x.size, reverse=True)
    
    print("Top 20 largest files:")
    print("\n".join([
        "%50s : %20s" % (m.path.split("/")[-1], format_file_size(m.size)) 
        for m in all_file_entries[:20]
    ]))
    

    dataset_name_to_class = {}
    for dataset in classification_datasets:
        dataset_name_to_class[dataset.path] = ClassificationDataset
    for dataset in forecasting_datasets_long:
        dataset_name_to_class[dataset.path] = LongForecastingDataset
    for dataset in forecasting_datasets_short:
        dataset_name_to_class[dataset.path] = ShortForecastingDataset
    for dataset in anomaly_detection_datasets:
        dataset_name_to_class[dataset.path] = AnomalyDetectionDataset
    
    dataset_name_to_class = _sample_datasets(dataset_name_to_class, datasets_fraction)
    
    # If max_files is specified, randomly select a subset of the datasets
    if max_files is not None and max_files > 0 and max_files < len(dataset_name_to_class):
        print(f"Selecting {max_files} random datasets from {len(dataset_name_to_class)} datasets")
        # Get all keys and shuffle them
        all_keys = list(dataset_name_to_class.keys())
        random.shuffle(all_keys)
        # Select only max_files keys
        selected_keys = all_keys[:max_files]
        # Create a new dict with only the selected keys
        dataset_name_to_class = {k: dataset_name_to_class[k] for k in selected_keys}
        
    datasets = list(dataset_name_to_class.keys())
    return datasets, dataset_name_to_class


def _get_labels(examples):
    labels = [example.labels for example in examples]
    labels = np.asarray(labels)
    return labels


def _get_forecasts(examples):
    forecasts = [torch.from_numpy(example.forecast) for example in examples]
    forecasts = torch.stack(forecasts)
    return forecasts


def _collate_fn_basic(examples):
    examples = list(filter(lambda x: x is not None, examples))
    timeseries = [torch.from_numpy(example.timeseries) for example in examples]
    input_masks = [torch.from_numpy(example.input_mask) for example in examples]
    names = [example.name for example in examples]
    timeseries = torch.stack(timeseries)
    input_masks = torch.stack(input_masks)
    names = np.asarray(names)

    return TimeseriesData(timeseries=timeseries, input_mask=input_masks, name=names)


def _collate_fn_classification(examples):
    batch = _collate_fn_basic(examples)
    batch.labels = _get_labels(examples)
    return batch


def _collate_fn_anomaly_detection(examples):
    batch = _collate_fn_basic(examples)
    batch.labels = _get_labels(examples)
    return batch


def _collate_fn_forecasting(examples):
    batch = _collate_fn_basic(examples)
    batch.forecast = _get_forecasts(examples)
    return batch

def get_timeseries_dataloader(args, **kwargs):
    # Pass max_files to get_all_datasets if it exists in args
    max_files = getattr(args, 'max_files', None)
    all_datasets, dataset_name_to_class = get_all_datasets(
        args.datasets_fraction, 
        max_files=max_files, 
        seed=args.seed
    )
    logging.debug(
        "dataset_names",
        args.dataset_names,
        type(args.dataset_names),
        args.dataset_names == "all",
    )
    
    if args.dataset_names == "all":
        assert (
            args.task_name == "pre-training"
        ), "Only pre-training task supports all datasets"
        args.dataset_names = all_datasets
        def init_dataset(name, cls):
            args.full_file_path_and_name = name
            return cls(**vars(args))
        dataset_classes = []
        
        dataset_classes = Parallel(n_jobs=args.num_workers, return_as="generator")(
            delayed(init_dataset)(name, cls)
            for name, cls in dataset_name_to_class.items()
        )
        dataset_loader = tqdm(dataset_classes, total=len(args.dataset_names), desc="Loading datasets")
        dataset_classes = [d for d in dataset_loader]
        dataset = ConcatDataset(
            [ds for ds in dataset_classes if ds.length_timeseries >= args.seq_len]
        )
        # dataset = ConcatDataset(dataset_classes)

    elif isinstance(args.dataset_names, str):
        args.full_file_path_and_name = args.dataset_names
        dataset = dataset_name_to_class[args.dataset_names](**vars(args))

    elif isinstance(args.dataset_names, list):
        assert (
            args.task_name == "pre-training"
        ), "Only pre-training task supports multiple datasets"
        dataset_classes = []
        dataset_classes = Parallel(n_jobs=args.num_workers, return_as="generator")(
            delayed(dataset_name_to_class[name])(**vars(args))
            for name in args.dataset_names
        )
        dataset_loader = tqdm(dataset_classes, total=len(args.dataset_names), desc="Loading datasets")
        dataset_classes = [d for d in dataset_loader]
        
        dataset = ConcatDataset(
            [ds for ds in dataset_classes if ds.length_timeseries >= args.seq_len]
        )
    else:
        raise NotImplementedError

    collate_fn_map = {
        "pre-training": _collate_fn_basic,
        "imputation": _collate_fn_basic,
        "classification": _collate_fn_classification,
        "long-horizon-forecasting": _collate_fn_forecasting,
        "short-horizon-forecasting": _collate_fn_forecasting,
        "anomaly-detection": _collate_fn_anomaly_detection,
    }

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn_map[args.task_name],
    )

    return dataloader
