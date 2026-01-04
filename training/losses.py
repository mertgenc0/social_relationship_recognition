"""
Loss Functions for Social Relationship Recognition
Includes Classification Loss + Contrastive Loss (from baseline paper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# models/baseline/losses.py dosyasını bu şekilde güncelle

class CombinedLoss(nn.Module):
    """
    Combined Loss = Classification Loss + Contrastive Loss

    From baseline paper:
    L_total = L_CE + α * L_contrastive

    where:
    - L_CE: Cross-entropy for relationship classification
    - L_contrastive: CLIP-style contrastive loss for multimodal alignment
    - α: weight for contrastive loss (default: 0.1)
    """

    def __init__(self, num_classes=6, alpha=0.1, label_smoothing=0.0):
        super(CombinedLoss, self).__init__()

        self.num_classes = num_classes
        self.alpha = alpha

        # Classification loss with optional label smoothing
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        print(f"🔧 Initializing Combined Loss...")
        print(f"   Number of classes: {num_classes}")
        print(f"   Contrastive weight (α): {alpha}")
        print(f"   Label smoothing: {label_smoothing}")

    def classification_loss(self, logits, labels):
        """
        Cross-entropy loss for relationship classification

        Args:
            logits: [batch_size, num_classes]
            labels: [batch_size]

        Returns:
            loss: scalar
        """
        return self.ce_loss(logits, labels)

    def contrastive_loss(self, similarity_matrix):
        """
        Contrastive loss for image-text alignment (CLIP-style)

        Args:
            similarity_matrix: [batch_size, batch_size]
                Cosine similarity between image and text features

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

    def forward(self, outputs, labels):
        """
        Compute total loss

        Args:
            outputs: dict from model forward pass containing:
                - logits: [batch_size, num_classes]
                - similarity_matrix: [batch_size, batch_size]
            labels: [batch_size] - ground truth labels

        Returns:
            total_loss: scalar
            loss_dict: dict with individual losses for logging
        """
        # Classification loss
        cls_loss = self.classification_loss(outputs['logits'], labels)

        # Contrastive loss
        cont_loss = self.contrastive_loss(outputs['similarity_matrix'])

        # Total loss
        total_loss = cls_loss + self.alpha * cont_loss

        # Return individual losses for logging
        loss_dict = {
            'total': total_loss.item(),
            'classification': cls_loss.item(),
            'contrastive': cont_loss.item(),
        }

        return total_loss, loss_dict


# Test code
# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing Loss Functions")
    print("=" * 60)

    # Create loss module
    loss_fn = CombinedLoss(num_classes=6, alpha=0.1, label_smoothing=0.1)

    # Test with sample data
    batch_size = 4
    num_classes = 6

    # Simulate model outputs (requires_grad=True for backward test)
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    similarity_matrix = torch.randn(batch_size, batch_size, requires_grad=True)
    labels = torch.randint(0, num_classes, (batch_size,))

    outputs = {
        'logits': logits,
        'similarity_matrix': similarity_matrix
    }

    print(f"\n📊 Test inputs:")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Similarity matrix shape: {similarity_matrix.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Labels: {labels}")

    # Compute loss
    print(f"\n⚙️  Computing loss...")
    total_loss, loss_dict = loss_fn(outputs, labels)

    print(f"\n✅ Loss computation successful!")
    print(f"\n📉 Loss values:")
    print(f"   Total loss: {loss_dict['total']:.4f}")
    print(f"   Classification loss: {loss_dict['classification']:.4f}")
    print(f"   Contrastive loss: {loss_dict['contrastive']:.4f}")

    # Test backward pass
    print(f"\n🔙 Testing backward pass...")
    total_loss.backward()
    print(f"   ✓ Backward pass successful!")
    print(f"   ✓ Gradients computed: logits.grad.shape = {logits.grad.shape}")

    # Test with perfect predictions
    print(f"\n🧪 Testing with perfect predictions...")
    perfect_logits = torch.zeros(batch_size, num_classes, requires_grad=True)
    perfect_labels = torch.arange(batch_size) % num_classes
    for i, label in enumerate(perfect_labels):
        perfect_logits.data[i, label] = 10.0  # High confidence for correct class

    perfect_outputs = {
        'logits': perfect_logits,
        'similarity_matrix': torch.eye(batch_size) * 10  # Perfect alignment
    }

    perfect_loss, perfect_loss_dict = loss_fn(perfect_outputs, perfect_labels)

    print(f"   Perfect prediction loss: {perfect_loss_dict['total']:.4f}")
    print(f"   (Should be close to 0)")

    # Test with random predictions
    print(f"\n🧪 Testing with random predictions...")
    random_outputs = {
        'logits': torch.randn(batch_size, num_classes) * 0.1,
        'similarity_matrix': torch.randn(batch_size, batch_size) * 0.1
    }

    random_loss, random_loss_dict = loss_fn(random_outputs, perfect_labels)

    print(f"   Random prediction loss: {random_loss_dict['total']:.4f}")
    print(f"   (Should be higher than perfect)")

    print("\n" + "=" * 60)
    print("✅ Loss Functions Test PASSED!")
    print("=" * 60)