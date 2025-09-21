import wandb
from os.path import join
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import defaultdict
from utils.networks import CBMNetwork
from utils.model_utils import unfreeze_module, freeze_module

class CBM:

    def __init__(self, args, train_loader, val_loader, test_loader, device):
        
        self.device = device
        self.model = CBMNetwork(args).to(self.device)
        self.args = args
        self.alpha = args.alpha if args.training_mode == "joint" else 1.0

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

    def reinitialize_optim(self):

        optim_params = [
            {
                "params": filter(lambda p: p.requires_grad, self.model.parameters()),
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
            }
        ]

        if self.args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(optim_params)
        elif self.args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(optim_params)

    def get_loss(self,
                pred_concepts: torch.Tensor,
                true_concepts: torch.Tensor,
                pred_targets: torch.Tensor,
                true_targets: torch.Tensor):
        
        assert torch.all((true_concepts == 0) | (true_concepts == 1))
        concepts_loss = self.compute_concept_loss(true_concepts, pred_concepts)

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

        total_loss = target_loss + concepts_loss

        return target_loss, concepts_loss, total_loss

    def compute_concept_loss(self, concepts_true, concepts_pred_probs):
        concepts_true = concepts_true.float() # [B, C]    
        concepts_loss = F.binary_cross_entropy(
            concepts_pred_probs, concepts_true , reduction='none'
        ) #  [B, C]
    
        if self.args.reduction == 'mean':
            concepts_loss = concepts_loss.mean(dim=0).sum()
        elif self.args.reduction == 'sum':
            concepts_loss = concepts_loss.sum(dim=0).sum()
    
        return (self.alpha * concepts_loss)
    
    def run(self):

        # For sequential & independent training: first stage is training of concept encoder
        if self.args.training_mode in ("sequential", "independent"):
            print("\nStarting concepts training!\n")
            mode = "c"

            # Freeze the target prediction part
            self.model.freeze_c()

            lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.args.decrease_every,
                gamma=1 / self.args.lr_divisor,
            )
            for epoch in range(self.args.c_epochs):
                # Validate the model periodically
                if epoch % self.args.validate_per_epoch == 0:
                    print("\nEVALUATION ON THE VALIDATION SET:\n")
                    self.validate_step(self.val_loader, epoch)
                self.train_step(mode, epoch)
                lr_scheduler.step()

                if self.args.save_model and epoch % 20 == 0:
                    torch.save(self.model.state_dict(), join(self.args.save_dir, "model.pth"))
                    print("SAVING THE MODEL !!!", flush=True)

            # Prepare parameters for target training by unfreezing the target prediction part and freezing the concept encoder
            self.model.freeze_t()

        self.reinitialize_optim()

        if self.args.training_mode in ("sequential", "independent"):
            print("\nStarting target training!\n")
            mode = "t"
        else:
            print("\nStarting joint training!\n")
            mode = "j"

        lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.args.decrease_every,
            gamma=1 / self.args.lr_divisor,
        )

        # If sequential & independent training: second stage is training of target predictor
        # If joint training: training of both concept encoder and target predictor
        for epoch in range(0, self.args.t_epochs):
            if epoch % self.args.validate_per_epoch == 0:
                print("\nEVALUATION ON THE VALIDATION SET:\n")
                self.validate_step(self.val_loader, epoch)
            self.train_step(mode, epoch)
            lr_scheduler.step()

            if self.args.save_model and epoch % 20 == 0:
                torch.save(self.model.state_dict(), join(self.args.save_dir, "model.pth"))
                print("SAVING THE MODEL !!!", flush=True)

        self.model.apply(freeze_module)
        if self.args.save_model:
            torch.save(self.model.state_dict(), join("./", "model.pth"))
            print("\nTRAINING FINISHED, MODEL SAVED!", flush=True)
        else:
            print("\nTRAINING FINISHED", flush=True)

        print("\nEVALUATION ON THE TEST SET:\n")
        self.validate_step(self.test_loader, epoch, test=True)

        if self.args.train_only:
            wandb.finish(quiet=True)

        # Intervention curves
        print("\nPERFORMING INTERVENTIONS:\n")

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


    def train_step(self, mode, epoch):

        self.model.train()

        if self.args.training_mode in ("sequential", "independent"):
            if mode == "c":
                self.model.head.eval()
            elif mode == "t":
                self.model.encoder.eval()

        metrics = defaultdict(list)
        total_correct = 0
        total_count = 0
        for k, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)):
            batch_features, target_true = batch["features"].to(self.device), batch["labels"].to(self.device)
            concepts_true = batch["concepts"].to(self.device)

            # Forward pass
            if self.args.training_mode == "independent" and mode == "t":
                concepts_pred_probs, target_pred_logits = self.model(batch_features)
            else:
                concepts_pred_probs, target_pred_logits = self.model(batch_features)
            
            # Backward pass depends on the training mode of the model
            self.optimizer.zero_grad()
            # Compute the loss
            target_loss, concepts_loss, total_loss = self.get_loss(concepts_pred_probs, concepts_true, target_pred_logits, target_true)

            if mode == "j":
                total_loss.backward()
            elif mode == "c":
                concepts_loss.backward()
            else:
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
            metrics["concepts_loss"].append(concepts_loss.item())
            metrics["total_loss"].append(total_loss.item())

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
                concepts_true = batch["concepts"].to(self.device)

                concepts_pred_probs, target_pred_logits = self.model(batch_features)
                target_loss, concepts_loss, total_loss = self.get_loss(concepts_pred_probs, concepts_true, target_pred_logits, target_true)

                preds, _ = self._preds_from_logits(target_pred_logits)
                if self.args.num_classes == 2:
                    labels = target_true.long()
                else:
                    labels = target_true.long()
                total_correct += (preds == labels).sum().item()
                total_count   += labels.numel()

                # Store predictions
                metrics["target_loss"].append(target_loss.item())
                metrics["concepts_loss"].append(concepts_loss.item())
                metrics["total_loss"].append(total_loss.item())

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
    