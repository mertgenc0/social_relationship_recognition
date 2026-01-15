"""
Image Encoder Module for Baseline Model
Uses ResNet-50 + Channel Attention + Spatial Attention

Görüntüden derin özellikler çıkarmak için ResNet-50 mimarisi kullanılır. Ancak sadece standart bir ResNet değil,
görüntünün önemli yerlerine odaklanan Channel (Kanal) ve Spatial (Mekansal) Attention (Dikkat)
mekanizmaları eklenmiştir.

Neden Uygulandı? Arka plandaki ağaçlar veya binalar yerine, sosyal ilişkiyi belirleyen
insan etkileşimlerine odaklanmak için
"""

import torch
import torch.nn as nn
import torchvision.models as models

class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    - Görüntünün hangi "özelliklerinin" (renk, doku, kenar vb.) daha önemli olduğuna karar verir.
    """

    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()

        # Global pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # Her kanalın genel özetini alır.
        self.max_pool = nn.AdaptiveMaxPool2d(1) # Her kanaldaki en belirgin özelliği yakalar.

        # Shared MLP (Çok katamnlı Algılayıcı)
        # Parametre sayısını azaltmak için kanal sayısını önce 16 ya böler (reduction), sonra eski haline getirir.
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()#s onucu 0 la 1 arasına sıkıştırır

    def forward(self, x):
        """
        Args:
            x: [batch, channels, height, width]
        Returns:
            attention: [batch, channels, 1, 1]

        - Ortalama ve maksimum yolları hesapla ardiında iki yolu topla ve sigmoidden geçirir
        çıktı her kanal için "Önem Skoru" olur.
        """
        batch, channels, _, _ = x.size() # girdi boyutu

        # Average pooling path
        avg_pool = self.avg_pool(x).view(batch, channels)
        avg_out = self.mlp(avg_pool)

        # Max pooling path
        max_pool = self.max_pool(x).view(batch, channels)
        max_out = self.mlp(max_pool)

        # Combine and sigmoid
        attention = self.sigmoid(avg_out + max_out).view(batch, channels, 1, 1)

        return attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    - Görüntünün hangi koordinatları (sağ üst, orta, sol alt vb.) ddaha önemli olduğunu belirlemek..
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        # Kanal bazında özetlenen veriyi işlemek için 7x7'lik büyük bir evrişim filtresi.
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: [batch, channels, height, width]
        Returns:
            attention: [batch, 1, height, width]

        -Kanlar üzerinde ilk ortalama sonra maksimum al bu iki haritayı üst üste ekle evrişim katamanı ile
        hani bölgenin önemli olduğunu öğren.
        """
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [batch, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [batch, 1, H, W]

        # Concatenate along channel dimension
        combined = torch.cat([avg_out, max_out], dim=1)  # [batch, 2, H, W]

        # Learn spatial attention
        attention = self.sigmoid(self.conv(combined))  # [batch, 1, H, W]

        return attention


class ResNetWithAttention(nn.Module):
    """
    ResNet-50 backbone with Channel and Spatial Attention

    Pipeline:
    1. ResNet-50 extracts features → [batch, 2048, 7, 7]
    2. Channel Attention weights channels → [batch, 2048, 1, 1]
    3. Spatial Attention weights spatial locations → [batch, 1, 7, 7]
    4. Combined attention applied to features
    5. Global average pooling → [batch, 2048]
    6. FC layer projects to hidden_dim → [batch, hidden_dim]

    -Önceden eğitilimiş Resnet-50 yi ve diğer iki dikakn mekanizmasını bir araya getirr.
    """

    def __init__(self, hidden_dim=256, pretrained=True):
        super(ResNetWithAttention, self).__init__()

        print(f"-  Initializing Image Encoder with ResNet-50...")

        # ResNet-50'yi hazır (ImageNet ile eğitilmiş) olarak indirir.
        resnet = models.resnet50(pretrained=pretrained)

        # ResNet'in katmanlarını tek tek parçalara ayırıyoruz:)
        # ResNet-50 structure: conv1 → bn1 → relu → maxpool → layer1-4 → avgpool → fc
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # ResNet 4 ana bloktan oluşur. Her blok görüntüyü daha derin işler.
        self.layer1 = resnet.layer1  # Output: 256 channels (Düşük seviyeli özellikler (çizgiler))
        self.layer2 = resnet.layer2  # Output: 512 channels
        self.layer3 = resnet.layer3  # Output: 1024 channels
        self.layer4 = resnet.layer4  # Output: 2048 channels (Yüksek seviyeli özellikler (nesne parçaları))

        # EĞİTİM STRATEJİSİ: İlk iki bloğu donduruyoruz. hızlı eğitim için ekelnemiştir güçlü bilgisayarda kadlır
        """
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
        """
        # Kendi dikkat modüllerimizi ekliyoruz
        self.channel_attention = ChannelAttention(2048, reduction=16)
        self.spatial_attention = SpatialAttention(kernel_size=7)

        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, hidden_dim)

        print(f"-  Image Encoder initialized")
        print(f"   Backbone: ResNet-50 (pretrained={pretrained})")
        print(f"   Feature dimension: 2048")
        print(f"   Output dimension: {hidden_dim}")

    def forward(self, x):
        """
        Args:
            x: [batch_size, 3, 224, 224] - input image

        Returns:
            image_features: [batch_size, hidden_dim]
        """
        # 1. ResNet-50 forward pass (standart ResNet işlemleri)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [batch, 2048, 7, 7]

        # 2. KANAL DİKKATİ: Önemli kanalları seç.
        # Çarpma işlemi (*) ile önemsiz kanalların değerlerini sıfıra yaklaştırır.
        channel_att = self.channel_attention(x)  # [batch, 2048, 1, 1]
        x = x * channel_att  # Broadcast multiply

        # 3. MEKANSAL DİKKAT: Önemli bölgeleri seç.
        # Çarpma işlemi ile arka planı karartıp, nesneye odaklanır
        spatial_att = self.spatial_attention(x)  # [batch, 1, 7, 7]
        x = x * spatial_att  # Broadcast multiply

        # Global pooling
        x= self.global_pool(x)  # [batch, 2048, 1, 1](Uzamsal boyutları yok et.)
        x= x.view(x.size(0), -1)  # [batch, 2048]

        # Project to hidden dimension(Son 256 boyutlu temsili üret.)
        image_features= self.fc(x)  # [batch, hidden_dim] (vektör halina getirir)

        return image_features

    def get_output_dim(self):
        """Return output feature dimension"""
        return self.fc.out_features


import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2 # Isı haritası renklendirmesi için

def visualize_model_focus(model, image_paths):
    """
    Belirlenen fotoğrafları modelden geçirir ve Spatial Attention
    (Mekansal Dikkat) haritalarını görselleştirir.
    """
    model.eval()

    # 1. Fotoğrafları modele hazırlama (Önişleme)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    plt.figure(figsize=(20, 10))

    for i, img_path in enumerate(image_paths):
        try:
            # Görüntüyü yükle
            raw_img = Image.open(img_path).convert('RGB')
            input_tensor = preprocess(raw_img).unsqueeze(0) # Batch boyutu ekle [1, 3, 224, 224]

            # 2. Modelden geçiş (Forward Pass)
            with torch.no_grad():
                # ResNet bloklarından geçerek layer4 çıktısını alalım
                x = model.maxpool(model.relu(model.bn1(model.conv1(input_tensor))))
                x = model.layer4(model.layer3(model.layer2(model.layer1(x))))

                # Sadece Spatial Attention haritasını çekelim
                attn_map = model.spatial_attention(x) # [1, 1, 7, 7]

                # Final özellik vektörünü de alalım (Görmek istersen)
                final_features = model(input_tensor)

            # 3. Dikkat haritasını işle (7x7'den 224x224'e büyütme)
            attn_map = attn_map.squeeze().cpu().numpy()
            attn_map_resized = cv2.resize(attn_map, (224, 224))

            # Normalize et (0-1 arası)
            attn_map_resized = (attn_map_resized - attn_map_resized.min()) / (attn_map_resized.max() - attn_map_resized.min())

            # 4. Görselleştirme
            # Orijinal Görüntü
            plt.subplot(2, len(image_paths), i + 1)
            plt.imshow(raw_img.resize((224, 224)))
            plt.title(f"Girdi: {img_path.split('/')[-1]}")
            plt.axis('off')

            # Isı Haritası (Dikkat Odakları)
            plt.subplot(2, len(image_paths), i + 1 + len(image_paths))
            plt.imshow(raw_img.resize((224, 224))) # Altlık olarak orijinal resim
            plt.imshow(attn_map_resized, cmap='jet', alpha=0.5) # Üstüne yarı saydam ısı haritası
            plt.title("Modelin Odaklandığı Yerler")
            plt.axis('off')

        except Exception as e:
            print(f"Hata: {img_path} yüklenemedi. {e}")

    plt.tight_layout()
    plt.show()

# --- GÜNCELLENMİŞ MAIN BLOĞU ---
if __name__ == "__main__":
    # Model oluşturuluyor
    my_model = ResNetWithAttention(hidden_dim=256, pretrained=True)

    # Kendi fotoğraflarının yollarını buraya yaz (Aynı klasördeyse direkt isimlerini yazabilirsin)
    # Örnek: ["arkadaslar.jpg", "aile.jpg", "ofis.jpg", "park.jpg"]
    my_test_images = ["data/dataset/image/00048.jpg", "data/dataset/image/00263.jpg", "data/dataset/image/00158.jpg", "data/dataset/image/08922.jpg"]

    print(f"\n- Seçilen {len(my_test_images)} fotoğraf üzerinde işlem başlatılıyor...")
    visualize_model_focus(my_model, my_test_images)






"""
if __name__ == "__main__":

    print(" -----Testing Image Encoder Module-------")

    # Create model
    model = ResNetWithAttention(hidden_dim=256, pretrained=True)
    model.eval()

    # Test with sample image
    print(f"\n- Creating test batch...")
    batch_size = 4
    test_images = torch.randn(batch_size, 3, 224, 224)
    print(f"   Input shape: {test_images.shape}")

    # Forward pass
    print(f"\n-  Running forward pass...")
    with torch.no_grad():
        features = model(test_images)

    print(f"\n-  Forward pass successful!")
    print(f"   Input: {batch_size} image [3, 224, 224]")
    print(f"   Output shape: {features.shape}")  # Should be [4, 256]
    print(f"   Output dtype: {features.dtype}")
    print(f"   Output range: [{features.min():.3f}, {features.max():.3f}]")

    # Test with single image
    print(f"\n🔍 Testing single image...")
    single_image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        single_feature = model(single_image)

    print(f"   Single output shape: {single_feature.shape}")  # Should be [1, 256]

    # Memory usage
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n- Model Statistics:")
    print(f"   Total parameters: {param_count:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {param_count - trainable_params:,}")

    # Test attention mechanisms separately
    print(f"\n- Testing Attention Mechanisms...")
    test_feature_map = torch.randn(2, 2048, 7, 7)

    channel_att = model.channel_attention(test_feature_map)
    print(f"   Channel attention output: {channel_att.shape}")  # [2, 2048, 1, 1]

    spatial_att = model.spatial_attention(test_feature_map)
    print(f"   Spatial attention output: {spatial_att.shape}")  # [2, 1, 7, 7]

    print(" -----Image Encoder Test PASSED!----- ")
"""





