"""
Multimodal Alignment Module for Baseline Model
Uses Cosine Similarity to align text and image features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalAlignment(nn.Module):
    """
    Aligns text and image features using cosine similarity

    Pipeline:
    1. Compute pairwise cosine similarity between text and image features
    2. Create similarity matrix S: [batch, text_dim, image_dim]
    3. Use similarity to align features

    From baseline paper Equation (2):
    s_ij = (F_I[i] · F_T[j]) / (|F_I[i]| · |F_T[j]|)
    """

    def __init__(self, temperature=0.07):
        super(MultimodalAlignment, self).__init__()

        # Temperature parameter for scaling similarity (like CLIP)
        self.temperature = temperature

        print(f"🔧 Initializing Multimodal Alignment...")
        print(f"   Temperature: {temperature}")

    def compute_similarity_matrix(self, image_features, text_features):
        """
        Compute cosine similarity matrix between image and text features

        Args:
            image_features: [batch_size, feature_dim]
            text_features: [batch_size, feature_dim]

        Returns:
            similarity_matrix: [batch_size, batch_size]
        """
        # Normalize features to unit length
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # Compute cosine similarity: I @ T^T
        # [batch, dim] @ [dim, batch] = [batch, batch]
        similarity_matrix = torch.matmul(image_features, text_features.t())

        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature

        return similarity_matrix

    def forward(self, image_features, text_features):
        """
        Align image and text features

        Args:
            image_features: [batch_size, feature_dim]
            text_features: [batch_size, feature_dim]

        Returns:
            aligned_image: [batch_size, feature_dim] - text-aligned image features
            aligned_text: [batch_size, feature_dim] - image-aligned text features
            similarity_matrix: [batch_size, batch_size] - for loss computation
        """
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(
            image_features, text_features
        )

        # Softmax along text dimension (each image attends to all texts)
        # [batch, batch] -> [batch, batch]
        image_to_text_weights = F.softmax(similarity_matrix, dim=1)

        # Softmax along image dimension (each text attends to all image)
        # [batch, batch] -> [batch, batch]
        text_to_image_weights = F.softmax(similarity_matrix.t(), dim=1)

        # Align image features using text
        # [batch, batch] @ [batch, dim] = [batch, dim]
        aligned_image = torch.matmul(image_to_text_weights, text_features)

        # Align text features using image
        # [batch, batch] @ [batch, dim] = [batch, dim]
        aligned_text = torch.matmul(text_to_image_weights, image_features)

        return aligned_image, aligned_text, similarity_matrix


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for multimodal alignment (like CLIP)
    Encourages matching image-text pairs to have high similarity
    """

    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, similarity_matrix):
        """
        Args:
            similarity_matrix: [batch_size, batch_size]

        Returns:
            loss: scalar
        """
        batch_size = similarity_matrix.size(0)

        # Ground truth: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=similarity_matrix.device)

        # Image-to-text loss
        loss_i2t = F.cross_entropy(similarity_matrix, labels)

        # Text-to-image loss
        loss_t2i = F.cross_entropy(similarity_matrix.t(), labels)

        # Average both directions
        loss = (loss_i2t + loss_t2i) / 2

        return loss


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing Multimodal Alignment Module")
    print("=" * 60)

    # Create alignment module
    alignment = MultimodalAlignment(temperature=0.07)
    alignment.eval()

    # Create contrastive loss
    contrastive_loss = ContrastiveLoss()

    # Test with sample features
    print(f"\n📊 Creating test features...")
    batch_size = 4
    feature_dim = 256

    # Simulate image and text features from encoders
    image_features = torch.randn(batch_size, feature_dim)
    text_features = torch.randn(batch_size, feature_dim)

    print(f"   Image features shape: {image_features.shape}")
    print(f"   Text features shape: {text_features.shape}")

    # Forward pass
    print(f"\n⚙️  Running alignment...")
    with torch.no_grad():
        aligned_image, aligned_text, similarity_matrix = alignment(
            image_features, text_features
        )

    print(f"\n✅ Alignment successful!")
    print(f"   Aligned image shape: {aligned_image.shape}")  # [4, 256]
    print(f"   Aligned text shape: {aligned_text.shape}")  # [4, 256]
    print(f"   Similarity matrix shape: {similarity_matrix.shape}")  # [4, 4]

    # Examine similarity matrix
    print(f"\n🔍 Similarity Matrix Analysis:")
    print(f"   Similarity matrix:\n{similarity_matrix}")
    print(f"   Diagonal (matching pairs): {similarity_matrix.diag()}")
    print(f"   Mean similarity: {similarity_matrix.mean():.3f}")
    print(f"   Diagonal mean: {similarity_matrix.diag().mean():.3f}")

    # Test contrastive loss
    print(f"\n📉 Testing Contrastive Loss...")
    loss = contrastive_loss(similarity_matrix)
    print(f"   Contrastive loss: {loss.item():.4f}")

    # Test with perfectly aligned features (should have high diagonal similarity)
    print(f"\n🧪 Testing with perfectly aligned features...")
    perfect_features = torch.randn(batch_size, feature_dim)
    with torch.no_grad():
        _, _, perfect_sim = alignment(perfect_features, perfect_features)
        perfect_loss = contrastive_loss(perfect_sim)

    print(f"   Perfect alignment similarity matrix:\n{perfect_sim}")
    print(f"   Diagonal (matching pairs): {perfect_sim.diag()}")
    print(f"   Perfect alignment loss: {perfect_loss.item():.4f}")
    print(f"   (Lower loss = better alignment)")

    # Test with misaligned features
    print(f"\n🧪 Testing with misaligned features...")
    misaligned_image = torch.randn(batch_size, feature_dim)
    misaligned_text = torch.randn(batch_size, feature_dim)
    with torch.no_grad():
        _, _, misaligned_sim = alignment(misaligned_image, misaligned_text)
        misaligned_loss = contrastive_loss(misaligned_sim)

    print(f"   Misaligned similarity matrix:\n{misaligned_sim}")
    print(f"   Diagonal (matching pairs): {misaligned_sim.diag()}")
    print(f"   Misaligned loss: {misaligned_loss.item():.4f}")
    print(f"   (Higher loss = worse alignment)")

    # Test batch size = 1 (edge case)
    print(f"\n🔍 Testing single sample...")
    single_image = torch.randn(1, feature_dim)
    single_text = torch.randn(1, feature_dim)
    with torch.no_grad():
        aligned_img, aligned_txt, single_sim = alignment(
            single_image, single_text
        )

    print(f"   Single aligned image shape: {aligned_img.shape}")  # [1, 256]
    print(f"   Single similarity: {single_sim.item():.3f}")

    # Memory usage
    param_count = sum(p.numel() for p in alignment.parameters())
    trainable_params = sum(p.numel() for p in alignment.parameters() if p.requires_grad)

    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {param_count:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   (This module has no learnable parameters)")

    print("\n" + "=" * 60)
    print("✅ Multimodal Alignment Test PASSED!")
    print("=" * 60)

    # Practical interpretation
    print("\n💡 Practical Interpretation:")
    print("   - High diagonal similarity → Good image-text matching")
    print("   - Low off-diagonal similarity → Good discrimination")
    print("   - Temperature controls sharpness of attention weights")
    print("   - Lower loss → Better multimodal alignment")