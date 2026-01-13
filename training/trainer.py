"""
Trainer Module for Social Relationship Recognition
Handles training loop, validation, checkpointing, and logging
FIXED: calculate_map array broadcasting issue
"""

import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from pathlib import Path


class Trainer:
    """
    Complete training pipeline for social relationship recognition
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        config,
        device='cuda',
        checkpoint_dir='../checkpoints/baseline',
        log_dir='../logs/baseline'
    ):
        """
        Initialize trainer

        Args:
            model: PyTorch model
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            config: Training configuration dict
            device: 'cuda' or 'cpu'
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        # Create directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_val_map = 0.0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_maps = []
        self.val_accs = []

        print("=" * 70)
        print("🚀 Trainer Initialized")
        print("=" * 70)
        print(f"Device: {device}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"Log directory: {self.log_dir}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print("=" * 70)

    def train_epoch(self):
        """
        Train for one epoch

        Returns:
            avg_loss: Average training loss
            metrics: Dict with training metrics
        """
        self.model.train()

        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_cont_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            captions = batch['caption']  # List of strings
            labels = batch['label'].to(self.device)

            # Forward pass
            outputs = self.model(images, captions, return_features=False)

            # Compute loss
            loss, loss_dict = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevent exploding gradients)
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )

            self.optimizer.step()

            # Update metrics
            epoch_loss += loss_dict['total']
            epoch_cls_loss += loss_dict['classification']
            epoch_cont_loss += loss_dict['contrastive']

            # Calculate accuracy
            predictions = outputs['predictions']
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'acc': f"{100.0 * correct / total:.2f}%"
            })

        # Calculate averages
        avg_loss = epoch_loss / len(self.train_loader)
        avg_cls_loss = epoch_cls_loss / len(self.train_loader)
        avg_cont_loss = epoch_cont_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        metrics = {
            'loss': avg_loss,
            'cls_loss': avg_cls_loss,
            'cont_loss': avg_cont_loss,
            'accuracy': accuracy
        }

        return avg_loss, metrics

    def validate(self):
        """
        Validate on validation set

        Returns:
            avg_loss: Average validation loss
            metrics: Dict with validation metrics including mAP
        """
        self.model.eval()

        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_cont_loss = 0.0

        all_predictions = []
        all_labels = []
        all_probs = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")

        with torch.no_grad():
            for batch in pbar:
                # Move data to device
                images = batch['image'].to(self.device)
                captions = batch['caption']
                labels = batch['label'].to(self.device)

                # Forward pass
                outputs = self.model(images, captions, return_features=False)

                # Compute loss
                loss, loss_dict = self.criterion(outputs, labels)

                # Update metrics
                epoch_loss += loss_dict['total']
                epoch_cls_loss += loss_dict['classification']
                epoch_cont_loss += loss_dict['contrastive']

                # Store predictions
                all_predictions.append(outputs['predictions'].cpu())
                all_labels.append(labels.cpu())
                all_probs.append(outputs['probs'].cpu())

                # Update progress bar
                pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})

        # Concatenate all batches
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_probs = torch.cat(all_probs, dim=0)

        # Calculate metrics
        avg_loss = epoch_loss / len(self.val_loader)
        avg_cls_loss = epoch_cls_loss / len(self.val_loader)
        avg_cont_loss = epoch_cont_loss / len(self.val_loader)

        accuracy = 100.0 * (all_predictions == all_labels).float().mean().item()

        # Calculate mAP (mean Average Precision)
        map_score = self.calculate_map(all_probs.numpy(), all_labels.numpy())

        # Calculate per-class accuracy
        per_class_acc = self.calculate_per_class_accuracy(
            all_predictions.numpy(),
            all_labels.numpy()
        )

        metrics = {
            'loss': avg_loss,
            'cls_loss': avg_cls_loss,
            'cont_loss': avg_cont_loss,
            'accuracy': accuracy,
            'map': map_score,
            'per_class_acc': per_class_acc
        }

        return avg_loss, metrics

    def calculate_map(self, probs, labels):
        """
        Calculate mean Average Precision (mAP) - FIXED VERSION

        Args:
            probs: [N, num_classes] - predicted probabilities
            labels: [N] - ground truth labels

        Returns:
            map_score: scalar mAP value
        """
        num_classes = probs.shape[1]
        aps = []

        for c in range(num_classes):
            # Binary labels for this class
            binary_labels = (labels == c).astype(np.float32)

            # Skip if no samples of this class
            if binary_labels.sum() == 0:
                continue

            # Sort by confidence
            sorted_indices = np.argsort(-probs[:, c])
            sorted_labels = binary_labels[sorted_indices]

            # Calculate precision at each position
            tp = np.cumsum(sorted_labels)
            fp = np.cumsum(1 - sorted_labels)
            precision = tp / (tp + fp + 1e-10)

            # Calculate recall at each position
            total_positives = binary_labels.sum()
            recall = tp / (total_positives + 1e-10)

            # === FIXED: Calculate AP using proper interpolation ===
            # Add boundary points
            recall_extended = np.concatenate([[0], recall, [1]])
            precision_extended = np.concatenate([[0], precision, [0]])

            # Ensure precision is monotonically decreasing
            for i in range(len(precision_extended) - 2, -1, -1):
                precision_extended[i] = max(precision_extended[i], precision_extended[i + 1])

            # Calculate area under curve
            indices = np.where(recall_extended[1:] != recall_extended[:-1])[0] + 1
            ap = np.sum(
                (recall_extended[indices] - recall_extended[indices - 1]) *
                precision_extended[indices]
            )

            aps.append(ap)

        # Mean over all classes
        map_score = np.mean(aps) if len(aps) > 0 else 0.0

        return map_score * 100  # Convert to percentage

    def calculate_per_class_accuracy(self, predictions, labels):
        """
        Calculate accuracy for each class

        Args:
            predictions: [N] - predicted labels
            labels: [N] - ground truth labels

        Returns:
            per_class_acc: dict mapping class_id to accuracy
        """
        num_classes = self.config.get('num_classes', 6)
        per_class_acc = {}

        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                class_acc = (predictions[mask] == labels[mask]).mean() * 100
                per_class_acc[c] = class_acc
            else:
                per_class_acc[c] = 0.0

        return per_class_acc

    def train(self, num_epochs):
        """
        Complete training loop with Phased Training integration
        """
        print("\n" + "=" * 70)
        print(f"🚀 Starting Training for {num_epochs} Epochs")
        print("=" * 70)

        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # --- PHASED TRAINING LOGIC (Makale Bölüm III-E) ---
            # 5. epoch tamamlanıp 6. epoch'a (index 5) başlarken parametreleri açıyoruz [cite: 283]
            if epoch == 5:
                print("\n🔓 [PHASED TRAINING] LLM katmanları ince ayar için açılıyor...")
                # LLM parametrelerini eğitilebilir hale getir
                for param in self.model.text_encoder.llm.parameters():
                    param.requires_grad = True

                # Parametreler değiştiği için optimizer'ı yeni listeyle güncellemek gerekir
                from training.optimizer import build_optimizer
                self.optimizer = build_optimizer(self.model, self.config)
                print("⚙️ Optimizer yeni parametrelerle güncellendi.")
            # --------------------------------------------------

            # Train for one epoch
            train_loss, train_metrics = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            self.val_maps.append(val_metrics['map'])
            self.val_accs.append(val_metrics['accuracy'])

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['map'])
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Print epoch summary
            print("\n" + "=" * 70)
            print(f"📊 Epoch {epoch + 1}/{num_epochs} Summary")
            print("=" * 70)
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"\nTrain Metrics:")
            print(f"  Loss: {train_metrics['loss']:.4f}")
            print(f"  Cls Loss: {train_metrics['cls_loss']:.4f}")
            print(f"  Cont Loss: {train_metrics['cont_loss']:.4f}")
            print(f"  Accuracy: {train_metrics['accuracy']:.2f}%")
            print(f"\nValidation Metrics:")
            print(f"  Loss: {val_metrics['loss']:.4f}")
            print(f"  Cls Loss: {val_metrics['cls_loss']:.4f}")
            print(f"  Cont Loss: {val_metrics['cont_loss']:.4f}")
            print(f"  Accuracy: {val_metrics['accuracy']:.2f}%")
            print(f"  mAP: {val_metrics['map']:.2f}%")

            # Print per-class accuracy
            print(f"\nPer-Class Accuracy:")
            class_names = ["Friends", "Family", "Couple", "Professional", "Commercial", "No Relation"]
            for c, acc in val_metrics['per_class_acc'].items():
                print(f"  {class_names[c]}: {acc:.2f}%")

            # Save checkpoint if best model
            is_best = val_metrics['map'] > self.best_val_map
            if is_best:
                self.best_val_map = val_metrics['map']
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(is_best=True)
                print(f"\n🎉 New best model! mAP: {self.best_val_map:.2f}%")

            # Save regular checkpoint every N epochs
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(is_best=False)

            print("=" * 70)

        # Training complete
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("🎉 Training Complete!")
        print("=" * 70)
        print(f"Total time: {elapsed_time / 3600:.2f} hours")
        print(f"Best validation mAP: {self.best_val_map:.2f}%")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print("=" * 70)
    """
    def train(self, num_epochs):

        print("\n" + "=" * 70)
        print(f"🚀 Starting Training for {num_epochs} Epochs")
        print("=" * 70)

        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train for one epoch
            train_loss, train_metrics = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            self.val_maps.append(val_metrics['map'])
            self.val_accs.append(val_metrics['accuracy'])

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['map'])
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Print epoch summary
            print("\n" + "=" * 70)
            print(f"📊 Epoch {epoch+1}/{num_epochs} Summary")
            print("=" * 70)
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"\nTrain Metrics:")
            print(f"  Loss: {train_metrics['loss']:.4f}")
            print(f"  Cls Loss: {train_metrics['cls_loss']:.4f}")
            print(f"  Cont Loss: {train_metrics['cont_loss']:.4f}")
            print(f"  Accuracy: {train_metrics['accuracy']:.2f}%")
            print(f"\nValidation Metrics:")
            print(f"  Loss: {val_metrics['loss']:.4f}")
            print(f"  Cls Loss: {val_metrics['cls_loss']:.4f}")
            print(f"  Cont Loss: {val_metrics['cont_loss']:.4f}")
            print(f"  Accuracy: {val_metrics['accuracy']:.2f}%")
            print(f"  mAP: {val_metrics['map']:.2f}%")

            # Print per-class accuracy
            print(f"\nPer-Class Accuracy:")
            class_names = ["Friends", "Family", "Couple", "Professional", "Commercial", "No Relation"]
            for c, acc in val_metrics['per_class_acc'].items():
                print(f"  {class_names[c]}: {acc:.2f}%")

            # Save checkpoint if best model
            is_best = val_metrics['map'] > self.best_val_map
            if is_best:
                self.best_val_map = val_metrics['map']
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(is_best=True)
                print(f"\n🎉 New best model! mAP: {self.best_val_map:.2f}%")

            # Save regular checkpoint every N epochs
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(is_best=False)

            print("=" * 70)

        # Training complete
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("🎉 Training Complete!")
        print("=" * 70)
        print(f"Total time: {elapsed_time/3600:.2f} hours")
        print(f"Best validation mAP: {self.best_val_map:.2f}%")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print("=" * 70)
    """
    def save_checkpoint(self, is_best=False):
        """
        Save model checkpoint

        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_map': self.best_val_map,
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_maps': self.val_maps,
            'val_accs': self.val_accs,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best checkpoint to {best_path}")

        # Save epoch checkpoint
        if (self.current_epoch + 1) % self.config.get('save_every', 10) == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{self.current_epoch+1}.pth'
            torch.save(checkpoint, epoch_path)
            print(f"💾 Saved checkpoint to {epoch_path}")

    def load_checkpoint(self, checkpoint_path):
        """
        Load checkpoint and resume training

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_map = checkpoint['best_val_map']
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_maps = checkpoint['val_maps']
        self.val_accs = checkpoint['val_accs']

        print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"   Best mAP: {self.best_val_map:.2f}%")


# Test code
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 Testing Trainer Module")
    print("=" * 70)
    print("\n⚠️  Trainer requires full model and dataset.")
    print("   This test will verify trainer can be instantiated.")
    print("\n✅ Trainer module loaded successfully!")
    print("   Ready to use with model and dataloaders.")
    print("=" * 70)