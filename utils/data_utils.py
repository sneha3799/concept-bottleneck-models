"""
Utility functions for data loading.
"""

import os
import torch
from torch.utils.data import DataLoader

from datasets import get_CUB_dataloaders

def get_data(args):
    """
    Parse the configuration file and return the relevant dataset loaders.

    This function parses the provided configuration file and returns the appropriate dataset loaders based on the
    specified dataset type. It also sets the data path based on the hostname or the configuration file if working
    locally and on a cluster. The function supports synthetic datasets, CUB, CIFAR-10, and CIFAR-100 datasets.

    Args:
        config_base (dict): The base configuration dictionary.
        config (dict): The data configuration dictionary containing dataset and data path information.
        gen (object): A generator object to control the randomness of the data loader.

    Returns:
        tuple: A tuple containing the training data loader, validation data loader, and test data loader.
    """

    if args.dataset == "CUB":
        print("Loading CUB Dataset")
        trainset, validset, testset = get_CUB_dataloaders(args)
    else:
        NotImplementedError("ERROR: Dataset not supported!")

    train_loader = DataLoader(
        trainset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        validset,
        batch_size=args.val_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        testset,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader, test_loader