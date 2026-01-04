import os
import torch
# Kendi yazdığın modülleri içe aktar
from data.pisc_dataset_loader import get_pisc_dataloaders
from models.baseline.baseline_model import BaselineModel
from training.losses import CombinedLoss
from training.optimizer import build_optimizer, build_scheduler
from training.trainer import Trainer

def main():
    # --- 1. Konfigürasyon ---
    config = {
        'data_root': 'data/dataset',  # Veri setinin yolu
        'num_classes': 6,             # Fine-grained (6 sınıf)
        'hidden_dim': 256,
        'batch_size': 4,              # M2 Mac belleği için ideal (BERT+ResNet ağır bir ikili)
        'lr': 1e-4,                   # Öğrenme hızı
        'num_epochs': 20,             # Toplam eğitim süresi
        'save_every': 5,
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu', # M2 GPU (Metal) desteği
        'checkpoint_dir': 'checkpoints/baseline',
        'log_dir': 'logs/baseline'
    }

    # Klasörleri oluştur
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    print(f"🚀 Baseline Eğitimi Başlıyor | Cihaz: {config['device'].upper()}")
    print(f"📊 Toplam Sınıf: {config['num_classes']} | Batch Size: {config['batch_size']}")

    # --- 2. DataLoaders (Çalıştığını teyit ettiğimiz loader) ---
    train_loader, val_loader = get_pisc_dataloaders(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=0 # M2 Mac'te stabilite için 0 kalmalı
    )

    # --- 3. Model Hazırlığı ---
    model = BaselineModel(
        num_classes=config['num_classes'],
        hidden_dim=config['hidden_dim'],
        pretrained_resnet=True
    ).to(config['device'])

    # --- 4. Loss, Optimizer ve Scheduler ---
    # L_total = L_CE + alpha * L_contrastive (Rapordaki formül)
    criterion = CombinedLoss(num_classes=config['num_classes'], alpha=0.1)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, num_epochs=config['num_epochs'])

    # --- 5. Trainer (Eğitim Döngüsü) ---
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=config['device'],
        checkpoint_dir=config['checkpoint_dir'],
        log_dir=config['log_dir']
    )

    # --- 6
    print("\n🎬 Eğitim döngüsü başlıyor...")
    trainer.train(num_epochs=config['num_epochs'])


if __name__ == "__main__":
    main()