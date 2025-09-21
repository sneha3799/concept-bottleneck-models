"""
Utility functions for data loading.
"""

import os
import torch
from torch.utils.data import DataLoader
from functools import partial

from datasets import get_CUB_dataloaders

def cbm_collate(batch, num_classes):
    # features: always float32 tensor
    feats = torch.stack([b["features"].to(torch.float32) for b in batch], dim=0)
    # concepts: always float32
    concs = torch.stack([torch.as_tensor(b["concepts"], dtype=torch.float32) for b in batch], dim=0)
    # labels: float32 for binary, int64 for multiclass
    if num_classes == 2:
        labs = torch.as_tensor([b["labels"] for b in batch], dtype=torch.float32)
    else:
        labs = torch.as_tensor([b["labels"] for b in batch], dtype=torch.long)
    return {"features": feats, "concepts": concs, "labels": labs}

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

    collate_fn = partial(cbm_collate, num_classes=getattr(args, "num_classes", 2))

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
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        validset,
        batch_size=args.val_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        testset,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader