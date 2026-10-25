
import random
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import config
from utilities.runtime import runtime
from torchvision.transforms import InterpolationMode


def dataset_info(dataset_config):
    name = dataset_config['dataset_name']
    classes = list(dataset_config['classes'])
    nfolds = int(dataset_config['number_of_folds'])
    train_df = pd.read_csv(config.TRAIN_FOLDS_DIRECTORY)
    test_df = pd.read_csv(config.TEST_DIRECTORY)
    
    runtime.TRAIN_DF = train_df
    runtime.TEST_DF = test_df
    runtime.DATASET_NAME = name
    runtime.CLASSES = classes
    runtime.NUMBER_OF_FOLDS = nfolds
    return train_df, test_df, name, classes, nfolds



def transform_setting():
    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=runtime.DERGEES,
            fill=0,
            translate=runtime.TRANSLATE, 
            scale=runtime.SCALE,
            interpolation = InterpolationMode.BILINEAR,
            ),
        transforms.ToTensor(),
        transforms.ColorJitter(
            brightness = runtime.BRIGHTNESS,
            contrast = runtime.CONTRAST,
        ),
        transforms.Normalize(
            mean = runtime.TRANSFORM_NORMALIZE_MEAN,
            std = runtime.TRANSFORM_NORMALIZE_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=runtime.TRANSFORM_NORMALIZE_MEAN,
                             std=runtime.TRANSFORM_NORMALIZE_STD),
    ])
    return train_transform, eval_transform



def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(dataset_class, classes, device, dataframe, transform, shuffle,
                      *, seed=None):
    dataset = dataset_class(dataframe=dataframe, classes=classes, transform=transform)
    generator = torch.Generator()
    generator.manual_seed(int(runtime.SEED if seed is None else seed))
    workers = int(runtime.NUMBBER_OF_WORKERS)
    loader = DataLoader(
        dataset, batch_size=int(runtime.BATCH_SIZE), shuffle=shuffle,
        num_workers=workers, prefetch_factor=2 if workers > 0 else None,
        pin_memory=(device.type == 'cuda'), drop_last=False,
        worker_init_fn=_seed_worker, generator=generator,
        persistent_workers=False,
    )
    return dataset, loader


def pos_weight(fold, device):
    path = (config.DATASET_DIRECTORY / 'pos_weight.npy' if int(fold) == 0
            else config.FOLDS_DIRECTORY / f'fold_{fold}' / 'pos_weight.npy')
    if not path.is_file():
        raise FileNotFoundError(f'Missing pos_weight: {path}')
    weights = np.load(path, allow_pickle=False)
    if weights.ndim != 1 or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError(f'Invalid pos_weight values in {path}')
    return torch.as_tensor(weights, dtype=torch.float32, device=device)
