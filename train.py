import argparse
import torch
from models import CBM, VanillaCNN
from utils.data_utils import get_data

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
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
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers to use for data loading")
    parser.add_argument("--save_model", type=bool, default=True, help="Whether to save the model")
    parser.add_argument("--model", type=str, default="CBM", help="Which model to train CBM/Vanilla CNN")

    # Dataset parameters
    parser.add_argument("--dataset_dir", type=str, default="./datasets", help="Which dataset to load")
    parser.add_argument("--dataset", type=str, default="CUB", help="Which dataset to load")

    # Model parameters
    parser.add_argument("--model_directory", type=str, default="./pretrained_networks/", help="Directory of pretrained models")
    parser.add_argument("--head_arch", type=str, default="linear", choices=["linear", "nonlinear"], help="Classifier head architecture")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight of concept loss in joint training")
    parser.add_argument("--encoder_arch", type=str, default="resnet18", choices=["resnet18", "simple_CNN"], help="Encoder backbone architecture")

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
    parser.add_argument("--j_epochs", type=int, default=30000, help="Epochs for joint training")
    parser.add_argument("--c_epochs", type=int, default=20000, help="Epochs for first stage training (sequential & independent)")
    parser.add_argument("--t_epochs", type=int, default=10000, help="Epochs for second stage training (sequential & independent)")
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

    