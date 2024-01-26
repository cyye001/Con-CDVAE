import random
from typing import Optional, Sequence
from pathlib import Path

import hydra
import numpy as np
import os
import omegaconf
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from concdvae.common.data_utils import get_scaler_from_data_list, get_maxAmin_from_data_list,GaussianDistance


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class CrystDataModule():
    def __init__(
        self,
        accelerator,
        n_delta,
        use_prop,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        scaler_path=None,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        train_path = self.datasets['train']['path']
        train_path = os.path.dirname(train_path)
        train_path = os.path.join(train_path, 'train_data.pt')
        if (os.path.exists(train_path)):
            self.train_dataset = torch.load(train_path)
        else:
            self.train_dataset = hydra.utils.instantiate(self.datasets.train, _recursive_=False)
            torch.save(self.train_dataset, train_path)
        print('load train')

        self.lattice_scaler = get_scaler_from_data_list(
            self.train_dataset.cached_data,
            key='scaled_lattice')

        val_path = self.datasets['val'][0]['path']
        val_path = os.path.dirname(val_path)
        val_path = os.path.join(val_path, 'val_data.pt')
        if (os.path.exists(val_path)):
            self.val_datasets = [torch.load(val_path)]
        else:
            self.val_datasets = [ hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.val]
            torch.save(self.val_datasets[0], val_path)
        print('load val')
        for val_dataset in self.val_datasets:
            val_dataset.lattice_scaler = self.lattice_scaler

        test_path = self.datasets['test'][0]['path']
        test_path = os.path.dirname(test_path)
        test_path = os.path.join(test_path, 'test_data.pt')
        if (os.path.exists(test_path)):
            self.test_datasets = [torch.load(test_path)]
        else:
            self.test_datasets = [hydra.utils.instantiate(dataset_cfg)
                                 for dataset_cfg in self.datasets.val]
            torch.save(self.test_datasets[0], test_path)
        print('load test')
        for test_dataset in self.test_datasets:
            test_dataset.lattice_scaler = self.lattice_scaler

        if accelerator == 'DDP':
            train_shuffle = False
            train_sampler = DistributedSampler(self.train_dataset)
            val_samplers = [DistributedSampler(dataset) for dataset in self.val_datasets]
            test_samplers = [DistributedSampler(dataset) for dataset in self.test_datasets]
        else:
            train_shuffle = True
            train_sampler = None
            val_samplers = [None for dataset in self.val_datasets]
            test_samplers = [None for dataset in self.test_datasets]


        self.train_dataloader = DataLoader(
            self.train_dataset,
            shuffle=train_shuffle,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
            sampler=train_sampler,
        )

        self.val_dataloaders = [
            DataLoader(
                self.val_datasets[i],
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                worker_init_fn=worker_init_fn,
                sampler=val_samplers[i]
            )
            for i in range(len(self.val_datasets))]

        self.test_dataloaders = [
            DataLoader(
                self.test_datasets[i],
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                worker_init_fn=worker_init_fn,
                sampler=test_samplers[i]
            )
            for i in range(len(self.test_datasets))]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )