"""
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np
from loguru import logger

from .base_dataset import BaseDataset
from ..utils.train_utils import parse_datasets_ratios


class MixedDataset(torch.utils.data.Dataset):

    def __init__(self, options, **kwargs):
        datasets_ratios = parse_datasets_ratios(options.DATASETS_AND_RATIOS)
        hl = len(datasets_ratios) // 2
        self.dataset_list = datasets_ratios[:hl]
        self.dataset_ratios = datasets_ratios[hl:]

        assert len(self.dataset_list) == len(self.dataset_ratios), 'Number of datasets and ratios should be equal'

        itw_datasets = ['mpii']

        self.datasets = [BaseDataset(options, ds, **kwargs)
                         for ds in self.dataset_list]
        length_itw = sum([len(ds) for ds in self.datasets if ds.dataset in itw_datasets])
        self.length = max([len(ds) for ds in self.datasets])

        self.partition = []

        for idx, (ds_name, ds_ratio) in enumerate(zip(self.dataset_list, self.dataset_ratios)):
            if ds_name in itw_datasets:
                r = ds_ratio * len(self.datasets[idx])/length_itw
            else:
                r = ds_ratio

            self.partition.append(r)

        logger.info(f'Using {self.dataset_list} dataset')
        logger.info(f'Ratios of datasets: {self.partition}')

        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(len(self.datasets)):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length
