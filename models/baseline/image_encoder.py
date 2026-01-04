"""
Image Encoder Module for Baseline Model
Uses ResNet-50 + Channel Attention + Spatial Attention
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    Learns to weight feature channels based on importance
    """

    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()

        # Global pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: [batch, channels, height, width]
        Returns:
            attention: [batch, channels, 1, 1]
        """
        batch, channels, _, _ = x.size()

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
    Learns to focus on important spatial regions
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        # Convolutional layer to learn spatial attention
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
    """

    def __init__(self, hidden_dim=256, pretrained=True):
        super(ResNetWithAttention, self).__init__()

        print(f"🔧 Initializing Image Encoder with ResNet-50...")

        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)

        # Remove the final FC layer (we'll add our own)
        # ResNet-50 structure: conv1 → bn1 → relu → maxpool → layer1-4 → avgpool → fc
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # Output: 256 channels
        self.layer2 = resnet.layer2  # Output: 512 channels
        self.layer3 = resnet.layer3  # Output: 1024 channels
        self.layer4 = resnet.layer4  # Output: 2048 channels

        # Freeze early layers (optional, saves memory)
        # Uncomment to freeze layer1 and layer2
        # for param in self.layer1.parameters():
        #     param.requires_grad = False
        # for param in self.layer2.parameters():
        #     param.requires_grad = False

        # Attention modules
        self.channel_attention = ChannelAttention(2048, reduction=16)
        self.spatial_attention = SpatialAttention(kernel_size=7)

        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, hidden_dim)

        print(f"✅ Image Encoder initialized")
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
        # ResNet-50 forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [batch, 2048, 7, 7]

        # Apply attention mechanisms
        # Channel attention
        channel_att = self.channel_attention(x)  # [batch, 2048, 1, 1]
        x = x * channel_att  # Broadcast multiply

        # Spatial attention
        spatial_att = self.spatial_attention(x)  # [batch, 1, 7, 7]
        x = x * spatial_att  # Broadcast multiply

        # Global pooling
        x = self.global_pool(x)  # [batch, 2048, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, 2048]

        # Project to hidden dimension
        image_features = self.fc(x)  # [batch, hidden_dim]

        return image_features

    def get_output_dim(self):
        """Return output feature dimension"""
        return self.fc.out_features


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing Image Encoder Module")
    print("=" * 60)

    # Create model
    model = ResNetWithAttention(hidden_dim=256, pretrained=True)
    model.eval()

    # Test with sample image
    print(f"\n📸 Creating test batch...")
    batch_size = 4
    test_images = torch.randn(batch_size, 3, 224, 224)
    print(f"   Input shape: {test_images.shape}")

    # Forward pass
    print(f"\n⚙️  Running forward pass...")
    with torch.no_grad():
        features = model(test_images)

    print(f"\n✅ Forward pass successful!")
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

    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {param_count:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {param_count - trainable_params:,}")

    # Test attention mechanisms separately
    print(f"\n🔍 Testing Attention Mechanisms...")
    test_feature_map = torch.randn(2, 2048, 7, 7)

    channel_att = model.channel_attention(test_feature_map)
    print(f"   Channel attention output: {channel_att.shape}")  # [2, 2048, 1, 1]

    spatial_att = model.spatial_attention(test_feature_map)
    print(f"   Spatial attention output: {spatial_att.shape}")  # [2, 1, 7, 7]

    print("\n" + "=" * 60)
    print("✅ Image Encoder Test PASSED!")
    print("=" * 60)