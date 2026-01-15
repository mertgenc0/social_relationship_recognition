import os
import torch
from data.pisc_dataset_loader_old import get_pisc_dataloaders
from models.baseline.baseline_model import BaselineModel
from training.losses import CombinedLoss
from training.optimizer import build_optimizer, build_scheduler
from training.trainer import Trainer

def main():
    config = {
        'use_enhanced': True,  # 🚀 TRUE: İnovasyonlar çalışır | FALSE: Makale ham hali (Baseline)
        'data_root': 'data/dataset',
        'num_classes': 6,
        'hidden_dim': 256,
        'batch_size': 128,
        'lr': 1e-3,
        'vocab_size': 1000,
        'embedding_dim': 128,
        'max_seq_length': 100,
        'num_epochs': 10,
        'save_every': 5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'checkpoint_dir': 'checkpoints/baseline',
        'log_dir': 'logs/baseline'
    }

    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    print(f"🚀 Baseline Eğitimi Başlıyor | Cihaz: {config['device'].upper()}")

    # --- GÜNCELLEME: 3 değer dönüyor (train, val, weights) ---
    train_loader, val_loader, test_loader, class_weights = get_pisc_dataloaders(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=4
    )

    model = BaselineModel(
        use_enhanced=config['use_enhanced'],
        num_classes=config['num_classes'],
        hidden_dim=config['hidden_dim'],
        pretrained_resnet=True
    ).to(config['device'])

    # --- GÜNCELLEME: Ağırlıkları cihaza gönder ve Loss'a ekle ---
    class_weights = class_weights.to(config['device'])
    criterion = CombinedLoss(
        use_enhanced=config['use_enhanced'],
        num_classes=config['num_classes'],
        alpha=0.1,
        weight=class_weights # Sınıf dengesizliği çözümü
    )

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, num_epochs=config['num_epochs'])

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

    print("\n🎬 Eğitim döngüsü başlıyor...")
    trainer.train(num_epochs=config['num_epochs'])

if __name__ == "__main__":
    main()