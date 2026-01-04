"""
Baseline Model Training Script - Optimized for M2 MacBook Air
Uses MPS (Metal Performance Shaders) for GPU acceleration
"""

import os
import sys
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.baseline.baseline_model import BaselineModel
from data.pisc_dataset_loader import get_pisc_dataloaders
from training.losses import CombinedLoss
from training.optimizer import build_optimizer, build_scheduler
from training.trainer import Trainer


def get_device():
    """
    Automatically detect best available device
    Priority: MPS (M1/M2) > CUDA > CPU
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU (slow)")

    return device


def main():
    print("=" * 70)
    print("🎓 Social Relationship Recognition - Baseline Training")
    print("🍎 Optimized for M2 MacBook Air")
    print("=" * 70)

    # ========== Configuration ==========
    config = {
        # Model
        'num_classes': 6,
        'hidden_dim': 256,
        'text_model_name': 'bert-base-uncased',
        'pretrained_resnet': True,

        # Training
        'batch_size': 4,  # M2 için optimize edilmiş
        'num_epochs': 20,  # İlk test için kısa
        'num_workers': 0,  # M2'de 0 en stabil

        # Optimizer
        'optimizer': 'adamw',
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'betas': (0.9, 0.999),

        # Scheduler
        'scheduler': 'cosine',
        'warmup_epochs': 2,
        'min_lr': 1e-6,

        # Loss
        'alpha': 0.1,  # Contrastive loss weight
        'label_smoothing': 0.1,

        # Regularization
        'grad_clip': 1.0,

        # Checkpointing
        'save_every': 5,

        # Paths
        'data_root': '../data/processed',
        'checkpoint_dir': '../checkpoints/baseline_m2',
        'log_dir': '../logs/baseline_m2'
    }

    # ========== Device Setup ==========
    device = get_device()

    # M2 için memory optimization
    if device.type == 'mps':
        print("\n🔧 M2 Optimizations:")
        print("   - Reduced batch size: 4")
        print("   - num_workers: 0 (most stable for MPS)")
        print("   - Mixed precision: disabled (MPS doesn't support yet)")

    # ========== Data Loading ==========
    print("\n📊 Loading Dataset...")
    train_loader, val_loader, test_loader = get_pisc_dataloaders(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    print(f"✅ Data loaded successfully!")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # ========== Model Creation ==========
    print("\n🏗️  Creating Model...")
    model = BaselineModel(
        num_classes=config['num_classes'],
        hidden_dim=config['hidden_dim'],
        text_model_name=config['text_model_name'],
        pretrained_resnet=config['pretrained_resnet']
    )

    # ========== Loss Function ==========
    print("\n📉 Setting up Loss Function...")
    criterion = CombinedLoss(
        num_classes=config['num_classes'],
        alpha=config['alpha'],
        label_smoothing=config['label_smoothing']
    )

    # ========== Optimizer & Scheduler ==========
    print("\n⚙️  Setting up Optimizer & Scheduler...")
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, config['num_epochs'])

    # ========== Trainer Setup ==========
    print("\n🚀 Initializing Trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        checkpoint_dir=config['checkpoint_dir'],
        log_dir=config['log_dir']
    )

    # ========== Memory Info ==========
    if device.type == 'mps':
        print("\n💾 Memory Estimate:")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Rough memory estimate (params + gradients + activations)
        model_memory = (total_params * 4) / (1024 ** 3)  # GB
        training_memory = model_memory * 3  # params + grads + activations

        print(f"   Model size: ~{model_memory:.2f} GB")
        print(f"   Training memory: ~{training_memory:.2f} GB")
        print(f"   Your M2 can handle this! ✅")

    # ========== Training ==========
    print("\n" + "=" * 70)
    print("🎯 Starting Training")
    print("=" * 70)

    try:
        trainer.train(num_epochs=config['num_epochs'])

        print("\n🎉 Training completed successfully!")
        print(f"💾 Best model saved to: {config['checkpoint_dir']}/best.pth")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("💾 Saving checkpoint...")
        trainer.save_checkpoint(is_best=False)
        print("✅ Checkpoint saved!")

    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("✅ Script completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()