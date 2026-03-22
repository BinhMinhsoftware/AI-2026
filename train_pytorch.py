"""
PyTorch Training script for Dog Breed Classification
"""
import os
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

import config_pytorch as config
from data_loader_pytorch import DogBreedDataLoader
from model_pytorch import create_model, get_optimizer, get_scheduler, count_parameters


class Trainer:
    """Training manager for dog breed classification"""

    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # Mixed precision training
        self.scaler = GradScaler() if config.USE_AMP else None

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_top5_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_top5_acc': [],
            'lr': []
        }

        # Best metrics
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        top5_correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{config.EPOCHS} [Train]")

        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass with mixed precision
            if config.USE_AMP:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, largest=True, sorted=True)
            top5_correct += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()

            # Update progress bar
            if batch_idx % config.LOG_INTERVAL == 0:
                pbar.set_postfix({
                    'loss': f'{running_loss / (batch_idx + 1):.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        epoch_top5_acc = 100. * top5_correct / total

        return epoch_loss, epoch_acc, epoch_top5_acc

    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        top5_correct = 0
        total = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{config.EPOCHS} [Val]")

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            if config.USE_AMP:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, largest=True, sorted=True)
            top5_correct += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss / (len(pbar)):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        epoch_top5_acc = 100. * top5_correct / total

        return epoch_loss, epoch_acc, epoch_top5_acc

    def train(self, num_epochs):
        """Main training loop"""
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning rate: {config.LEARNING_RATE}")
        print(f"Mixed precision: {config.USE_AMP}")
        print("=" * 80 + "\n")

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss, train_acc, train_top5_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc, val_top5_acc = self.validate(epoch)

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_top5_acc'].append(train_top5_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_top5_acc'].append(val_top5_acc)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train Top-5: {train_top5_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Top-5: {val_top5_acc:.2f}%")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f} | Time: {epoch_time:.1f}s")

            # Update learning rate scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_acc)
            else:
                self.scheduler.step()

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                self.save_checkpoint('best_model.pth', epoch, val_acc)
                print(f"  [BEST] New best model! Val Acc: {val_acc:.2f}%")
            else:
                self.epochs_no_improve += 1

            # Save periodic checkpoint
            if epoch % config.SAVE_CHECKPOINT_EVERY == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', epoch, val_acc)

            # Early stopping
            if self.epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n[WARNING] Early stopping triggered after {epoch} epochs")
                print(f"Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
                break

            print("-" * 80)

        # Training complete
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED!")
        print("=" * 80)
        print(f"Total training time: {total_time / 60:.1f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
        print("=" * 80 + "\n")

        # Save final model
        self.save_checkpoint('final_model.pth', num_epochs, val_acc)

        # Plot training history
        self.plot_history()

        return self.history

    def save_checkpoint(self, filename, epoch, val_acc):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(config.MODEL_DIR, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }, checkpoint_path)

    def plot_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy
        axes[0, 0].plot(self.history['train_acc'], label='Train Acc', marker='o')
        axes[0, 0].plot(self.history['val_acc'], label='Val Acc', marker='s')
        axes[0, 0].set_title('Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Loss
        axes[0, 1].plot(self.history['train_loss'], label='Train Loss', marker='o')
        axes[0, 1].plot(self.history['val_loss'], label='Val Loss', marker='s')
        axes[0, 1].set_title('Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Top-5 Accuracy
        axes[1, 0].plot(self.history['train_top5_acc'], label='Train Top-5', marker='o')
        axes[1, 0].plot(self.history['val_top5_acc'], label='Val Top-5', marker='s')
        axes[1, 0].set_title('Top-5 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-5 Accuracy (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Learning Rate
        axes[1, 1].plot(self.history['lr'], marker='o', color='red')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plot_path = os.path.join(config.RESULTS_DIR, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training history plot saved to: {plot_path}")


def main():
    """Main training function"""
    # Set random seeds
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)

    # Create data loaders
    print("\n[1/4] Loading data...")
    data_loader = DogBreedDataLoader()
    train_loader, val_loader, test_loader = data_loader.create_dataloaders(use_cache=True)
    data_loader.save_label_mapping()

    # Create model
    print("\n[2/4] Creating model...")
    model = create_model(
        architecture=config.MODEL_ARCHITECTURE,
        num_classes=config.NUM_CLASSES,
        pretrained=config.USE_PRETRAINED,
        freeze_backbone=config.FREEZE_BACKBONE
    )
    model = model.to(config.DEVICE)

    print("\nModel Architecture:", config.MODEL_ARCHITECTURE)
    count_parameters(model)

    # Create loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    # Create optimizer
    optimizer = get_optimizer(model, config.OPTIMIZER, config.LEARNING_RATE)

    # Create scheduler
    scheduler = get_scheduler(optimizer, 'reduce_on_plateau')

    # Create trainer
    print("\n[3/4] Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.DEVICE
    )

    # Train model
    print("\n[4/4] Training model...")
    history = trainer.train(config.EPOCHS)

    print("\n✓ Training complete! Model saved to:", config.MODEL_DIR)


if __name__ == "__main__":
    main()
