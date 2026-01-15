import torch
from models.baseline.baseline_model import BaselineModel


def test_modes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4

    # Test verisi oluştur
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_captions = ["relationship: friends, emotion: happy, setting: outdoor"] * batch_size
    dummy_labels = torch.randint(0, 6, (batch_size,)).to(device)

    for mode in [False, True]:
        mode_str = "ENHANCED" if mode else "BASELINE"
        print(f"\n--- Testing {mode_str} Mode ---")

        # Modeli yükle
        model = BaselineModel(use_enhanced=mode, num_classes=6).to(device)
        model.eval()

        # Forward pass
        with torch.no_grad():
            outputs = model(dummy_images, dummy_captions)

        # Çıktı kontrolleri
        print(f"✅ Logits shape: {outputs['logits'].shape}")  # [4, 6] olmalı

        # Loss hesaplama testi
        loss, loss_dict = model.compute_loss(outputs, dummy_labels)
        print(f"✅ Total Loss: {loss.item():.4f}")

        if mode and outputs['uncertainty'] is not None:
            print(f"✅ Uncertainty detected: {outputs['uncertainty'][0].shape}")


if __name__ == "__main__":
    test_modes()