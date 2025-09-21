import time
import uuid
import argparse
import os
from os.path import join
from pathlib import Path

import torch
import wandb
from models import CBM, VanillaCNN
from utils.data_utils import get_data

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    ex_name = "{}_{}".format(str(timestr), uuid.uuid4().hex[:5])
    experiment_path = (
        Path(args.save_dir) / args.model / args.dataset / ex_name
    )
    experiment_path.mkdir(parents=True)
    args.save_dir = str(experiment_path)
    print("Experiment path: ", experiment_path)

    wandb.init(
        project=args.project,
        reinit=True,
        entity=args.entity,
        mode=args.mode,
        tags=[args.tag],
    )
    if args.mode in ["online", "disabled"]:
        wandb.run.name = wandb.run.name.split("-")[-1] + "-" + args.exp_name
    elif args.mode == "offline":
        wandb.run.name = args.exp_name
    else:
        raise ValueError("wandb needs to be set to online, offline or disabled.")
    
    train_loader, val_loader, test_loader = get_data(args)

    if args.model == "CBM":
        algorithm = CBM(args, train_loader, val_loader, test_loader, device)
    else:
        algorithm = VanillaCNN(args, train_loader, val_loader, test_loader, device)
    
    algorithm.run()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CBM parameters")

    # Default parameters
    parser.add_argument("--exp_name", type=str, default="cbm_train", help="Name of the experiment")
    parser.add_argument("--save_dir", type=str, default="./cbm_train", help="Name of the directory to save the models")
    parser.add_argument("--seed", type=int, default=0, help="seed for training")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers to use for data loading")
    parser.add_argument("--save_model", type=bool, default=True, help="Whether to save the model")
    parser.add_argument("--train_only", type=bool, default=False, help="Whether to train the model only and exit before performing interventions")
    parser.add_argument("--model", type=str, default="VanillaCNN", help="Which model to train CBM/Vanilla CNN")

    # WandB parameters
    parser.add_argument("--project", type=str, default="CBM", help="Name of the wandb project")
    parser.add_argument("--mode", type=str, default="online", help="Whether to log to wandb")
    parser.add_argument("--entity", type=str, default="", help="WandB entity")
    parser.add_argument("--tag", type=str, default="baseline", help="Model's tag for wandb logging")

    # CBM parameters
    parser.add_argument("--concept_learning", type=str, default="hard", help="Concept Bottleneck Model with either hard {0,1} concepts or soft logit representations")

    # Dataset parameters
    parser.add_argument("--dataset_dir", type=str, default="./datasets", help="Which dataset to load")
    parser.add_argument("--dataset", type=str, default="CUB", help="Which dataset to load")

    # Model parameters
    parser.add_argument("--model_directory", type=str, default="./pretrained_networks/", help="Directory of pretrained models")
    parser.add_argument("--head_arch", type=str, default="linear", choices=["linear", "nonlinear"], help="Classifier head architecture")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight of concept loss in joint training")
    parser.add_argument("--encoder_arch", type=str, default="simple_CNN", choices=["resnet18", "simple_CNN"], help="Encoder backbone architecture")

    # Training parameters
    parser.add_argument("--training_mode", type=str, default="joint", choices=["joint", "sequential", "independent"], help="Optimization method")
    parser.add_argument("--validate_per_epoch", type=int, default=30, help="Periodicity to evaluate the model")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam"], help="Optimizer")
    parser.add_argument("--decrease_every", type=int, default=150, help="Epoch frequency to decrease learning rate")
    parser.add_argument("--lr_divisor", type=int, default=2, help="Factor to divide LR when decayed")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for optimizer")
    parser.add_argument("--train_batch_size", type=int, default=64, help="Batch size for training set")
    parser.add_argument("--val_batch_size", type=int, default=64, help="Batch size for validation/test sets")
    parser.add_argument("--j_epochs", type=int, default=300, help="Epochs for joint training")
    parser.add_argument("--c_epochs", type=int, default=200, help="Epochs for first stage training (sequential & independent)")
    parser.add_argument("--t_epochs", type=int, default=100, help="Epochs for second stage training (sequential & independent)")
    parser.add_argument("--reduction", type=str, default="mean", help="")

    args = parser.parse_args()

    # Dataset-specific overrides
    dataset_config = {
        "CUB":     {"num_classes": 200, "num_concepts": 112},
    }

    if args.dataset in dataset_config:
        for k, v in dataset_config[args.dataset].items():
            setattr(args, k, v)

    main(args)

    