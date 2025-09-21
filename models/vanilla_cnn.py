import wandb
from os.path import join
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import defaultdict
from utils.networks import VanillaCNNNetwork
from utils.model_utils import unfreeze_module, freeze_module

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

        self.model.apply(freeze_module)
        if self.args.save_model:
            torch.save(self.model.state_dict(), join("./", "model.pth"))
            print("\nTRAINING FINISHED, MODEL SAVED!", flush=True)
        else:
            print("\nTRAINING FINISHED", flush=True)

        print("\nEVALUATION ON THE TEST SET:\n")
        self.validate_step(self.test_loader, epoch, test=True)

        wandb.finish(quiet=True)

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
        wandb.log(log_payload)

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

        wandb.log(log_payload)
    