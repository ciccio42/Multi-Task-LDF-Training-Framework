import os
import random
from multiprocessing import cpu_count

import numpy as np
import torch
from hydra.utils import instantiate
from torch_geometric.loader import DataLoader
from torch.utils.data._utils.collate import default_collate
from multi_task_il_gnn.datasets.batch_sampler import BatchSampler
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
import time
from collections import defaultdict
import logging

colorama_init()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger object
logger = logging.getLogger('Data-Loader')


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_by_task(batch):
    """ Use this for validation: groups data by task names to compute per-task losses """
    collate_time = time.time()
    per_task_data = defaultdict(list)
    start_batch = time.time()
    for b in batch:
        per_task_data[b['task_name']].append(
            {k: v for k, v in b.items() if k != 'task_name' and k != 'task_id'}
        )
    logger.debug(f"Batch time {time.time()-start_batch}")

    collate_time = time.time()
    for name, data in per_task_data.items():
        per_task_data[name] = default_collate(data)
    logger.debug(f"Collate time {time.time()-collate_time}")
    return per_task_data


def make_data_loaders(config, dataset_cfg):

    print(f"{Fore.GREEN}Creating Trainign dataset{Style.RESET_ALL}")

    dataset_cfg.mode = "train"
    dataset = instantiate(dataset_cfg)

    train_step = int(config.get('epochs') *
                     int(len(dataset)/config.get('bsize')))
    print(f"{Fore.GREEN}Number of train/step {train_step}{Style.RESET_ALL}")

    samplerClass = BatchSampler
    train_sampler = samplerClass(
        task_to_idx=dataset.task_to_idx,
        subtask_to_idx=dataset.subtask_to_idx,
        tasks_spec=dataset_cfg.tasks_spec,
        sampler_spec=config.samplers,
        n_step=train_step)

    train_loader = DataLoader(
        dataset,
        batch_sampler=train_sampler,
        num_workers=config.get('loader_workers', cpu_count()),
        worker_init_fn=lambda w: np.random.seed(
            np.random.randint(2 ** 29) + w),
        collate_fn=collate_by_task,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    print(f"{Fore.GREEN}Creating Validation dataset{Style.RESET_ALL}")
    dataset_cfg.mode = 'val'
    val_dataset = instantiate(dataset_cfg)
    val_step = int(config.get('epochs') *
                   int(len(val_dataset)/config.get('vsize')))
    print(f"{Fore.GREEN}Number of val/step {val_step}{Style.RESET_ALL}")

    samplerClass = BatchSampler
    val_sampler = samplerClass(
        task_to_idx=val_dataset.task_to_idx,
        subtask_to_idx=val_dataset.subtask_to_idx,
        tasks_spec=dataset_cfg.tasks_spec,
        sampler_spec=config.samplers,
        n_step=val_step)
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=config.get('loader_workers', cpu_count()),
        worker_init_fn=lambda w: np.random.seed(
            np.random.randint(2 ** 29) + w),
        collate_fn=collate_by_task,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=True
    )

    return train_loader, val_loader
