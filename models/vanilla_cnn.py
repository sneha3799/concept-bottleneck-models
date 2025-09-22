import os
from tqdm import tqdm
from os.path import join

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models

from collections import defaultdict
from utils.model_utils import unfreeze_module, freeze_module, Identity

class VanillaCNNNetwork(nn.Module):

    def __init__(self, 
                args):

        super(VanillaCNNNetwork, self).__init__()

        self.args = args

        self.num_classes = args.num_classes
        self.encoder_arch = args.encoder_arch
        self.head_arch = args.head_arch

        self.setup(args)

    def setup(self, args):

        if self.encoder_arch == "resnet18":
            self.encoder_res = models.resnet18(weights=None)
            self.encoder_res.load_state_dict(
                torch.load(
                    os.path.join(
                        args.model_directory, "resnet/resnet18-5c106cde.pth"
                    ),
                    weights_only=False
                )
            )
            n_features = self.encoder_res.fc.in_features
            self.encoder_res.fc = Identity()
            self.encoder = nn.Sequential(self.encoder_res)

        elif self.encoder_arch == "simple_CNN":
            n_features = 256
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 5, 3),
                nn.ReLU(),
                nn.Conv2d(32, 64, 5, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
                nn.Flatten(),
                nn.Linear(9216, n_features),
                nn.ReLU(),
            )
        else:
            raise NotImplementedError("ERROR: architecture not supported!")

        if self.num_classes == 2:
            self.pred_dim = 1
        elif self.num_classes > 2:
            self.pred_dim = self.num_classes

        if self.head_arch == "linear":
            fc_y = nn.Linear(n_features, self.pred_dim)
            self.head = nn.Sequential(fc_y)
        else:
            fc1_y = nn.Linear(n_features, 256)
            fc2_y = nn.Linear(256, self.pred_dim)
            self.head = nn.Sequential(fc1_y, nn.ReLU(), fc2_y)

    def forward(self, x):

        intermediate = self.encoder(x)
        y_pred_logits = self.head(intermediate)

        return y_pred_logits
    
    def freeze_c(self):
        self.head.apply(freeze_module)

    def freeze_t(self):
        self.head.apply(unfreeze_module)
        self.encoder.apply(freeze_module)

class VanillaCNN:

    def __init__(self, args, train_loader, val_loader, test_loader, device):
        
        self.device = device
        self.model = VanillaCNNNetwork(args).to(self.device)
        self.args = args

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

        optim_params = [
            {
                "params": filter(lambda p: p.requires_grad, self.model.parameters()),
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            }
        ]

        if args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(optim_params)
        elif args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(optim_params)

    def get_loss(self,
                pred_targets: torch.Tensor,
                true_targets: torch.Tensor):
        
        if self.args.num_classes == 2:
            # Logits to probs
            target_pred_probs = nn.Sigmoid()(pred_targets.squeeze(1))
            target_loss = F.binary_cross_entropy(
                target_pred_probs, true_targets.float(), reduction=self.args.reduction
            )
        else:
            target_loss = F.cross_entropy(
                pred_targets, true_targets.long(), reduction=self.args.reduction
            )

        return target_loss
    
    def run(self):

        lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.args.decrease_every,
            gamma=1 / self.args.lr_divisor,
        )

        for epoch in range(0, self.args.t_epochs):
            if epoch % self.args.validate_per_epoch == 0:
                print("\nEVALUATION ON THE VALIDATION SET:\n")
                self.validate_step(self.val_loader, epoch)
            self.train_step(epoch)
            lr_scheduler.step()

            if self.args.save_model and epoch % 20 == 0:
                torch.save(self.model.state_dict(), join(self.args.save_dir, "model.pth"))
                print("SAVING THE MODEL !!!", flush=True)

        self.model.apply(freeze_module)
        if self.args.save_model:
            torch.save(self.model.state_dict(), join(self.args.save_dir, "model.pth"))
            print("\nTRAINING FINISHED, MODEL SAVED!", flush=True)
        else:
            print("\nTRAINING FINISHED", flush=True)

        print("\nEVALUATION ON THE TEST SET:\n")
        self.validate_step(self.test_loader, epoch, test=True)

    def _preds_from_logits(self, logits: torch.Tensor):
        """
        Returns (preds, probs) where:
        - binary: preds in {0,1}, probs is sigmoid probability (B,)
        - multiclass: preds in {0..K-1}, probs is softmax (B,K)
        """
        if self.args.num_classes == 2:
            probs = torch.sigmoid(logits.squeeze(1))      # (B,)
            preds = (probs >= 0.5).long()                 # (B,)
            return preds, probs
        else:
            probs = torch.softmax(logits, dim=1)          # (B, K)
            preds = probs.argmax(dim=1)                   # (B,)
            return preds, probs


    def train_step(self, epoch):

        self.model.train()

        metrics = defaultdict(list)
        total_correct = 0
        total_count = 0
        for k, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)):
            batch_features, target_true = batch["features"].to(self.device), batch["labels"].to(self.device)

            # Forward pass
            target_pred_logits = self.model(batch_features)
            
            # Backward pass depends on the training mode of the model
            self.optimizer.zero_grad()
            # Compute the loss
            target_loss = self.get_loss(target_pred_logits, target_true)
            target_loss.backward()
            self.optimizer.step()  # perform an update

            preds, _ = self._preds_from_logits(target_pred_logits)
            if self.args.num_classes == 2:
                labels = target_true.long()
            else:
                labels = target_true.long()
            total_correct += (preds == labels).sum().item()
            total_count   += labels.numel()


            # Store predictions
            metrics["target_loss"].append(target_loss.item())

        metrics_mean = {k: sum(v) / len(v) for k, v in metrics.items() if len(v) > 0}
        acc = total_correct / max(1, total_count)

        # --- build log dict with dataset info ---
        log_payload = {
            "epoch": epoch + 1,
            **{f"train/{k}": v for k, v in metrics_mean.items()},
            "train/acc": acc
        }
        print(log_payload)

    def validate_step(self, loader, epoch, test=False):

        self.model.eval()
        metrics = defaultdict(list)
        total_correct = 0
        total_count = 0
        with torch.no_grad():
            for k, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}", position=0, leave=True)):
                
                batch_features, target_true = batch["features"].to(self.device), batch["labels"].to(self.device)

                target_pred_logits = self.model(batch_features)
                target_loss = self.get_loss(target_pred_logits, target_true)
                
                preds, _ = self._preds_from_logits(target_pred_logits)
                if self.args.num_classes == 2:
                    labels = target_true.long()
                else:
                    labels = target_true.long()
                total_correct += (preds == labels).sum().item()
                total_count   += labels.numel()

                # Store predictions
                metrics["target_loss"].append(target_loss.item())

        metrics_mean = {k: sum(v) / len(v) for k, v in metrics.items() if len(v) > 0}
        acc = total_correct / max(1, total_count)
        if test:
            # --- build log dict with dataset info ---
            log_payload = {
                "epoch": epoch + 1,
                **{f"test/{k}": v for k, v in metrics_mean.items()},
                "test/acc": acc,
            }
        else:
            # --- build log dict with dataset info ---
            log_payload = {
                "epoch": epoch + 1,
                **{f"validation/{k}": v for k, v in metrics_mean.items()},
                "validation/acc": acc,
            }

        print(log_payload)
    