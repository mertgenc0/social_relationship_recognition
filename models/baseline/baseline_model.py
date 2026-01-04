"""
Complete Baseline Model for Social Relationship Recognition
Combines all components: Text Encoder + Image Encoder + Alignment + Fusion + Classifier
"""

import torch
import torch.nn as nn
from .text_encoder import LLMTextEncoder
from .image_encoder import ResNetWithAttention
from .alignment import MultimodalAlignment, ContrastiveLoss
from .fusion import AdaptiveFusion
from .classifier import RelationshipClassifier


class BaselineModel(nn.Module):
    """
    Complete Baseline Model

    Architecture Pipeline:
    1. Text Encoder: LLM (BERT) + CNN → text_features [batch, 256]
    2. Image Encoder: ResNet-50 + Attention → image_features [batch, 256]
    3. Alignment: Cosine similarity alignment → aligned_image, aligned_text
    4. Fusion: Adaptive weighted fusion → fused_features [batch, 256]
    5. Classifier: Multi-layer MLP → predictions [batch, num_classes]

    From baseline paper: "Multimodal Social Relationship Recognition" (AAAI 2022)
    """

    def __init__(
            self,
            num_classes=6,
            hidden_dim=256,
            text_model_name='bert-base-uncased',
            pretrained_resnet=True,
            alignment_temperature=0.07,
            fusion_hidden_dim=128,
            classifier_hidden_dim=128,
            classifier_dropout=0.3
    ):
        super(BaselineModel, self).__init__()

        print("=" * 70)
        print("🚀 Initializing Complete Baseline Model")
        print("=" * 70)

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # 1. Text Encoder
        print("\n[1/5] Initializing Text Encoder...")
        self.text_encoder = LLMTextEncoder(
            model_name=text_model_name,
            hidden_dim=hidden_dim
        )

        # 2. Image Encoder
        print("\n[2/5] Initializing Image Encoder...")
        self.image_encoder = ResNetWithAttention(
            hidden_dim=hidden_dim,
            pretrained=pretrained_resnet
        )

        # 3. Multimodal Alignment
        print("\n[3/5] Initializing Multimodal Alignment...")
        self.alignment = MultimodalAlignment(temperature=alignment_temperature)
        print(f"✅ Multimodal Alignment initialized")

        # 4. Fusion Module
        print("\n[4/5] Initializing Fusion Module...")
        self.fusion = AdaptiveFusion(
            feature_dim=hidden_dim,
            hidden_dim=fusion_hidden_dim
        )

        # 5. Classifier
        print("\n[5/5] Initializing Classifier...")
        self.classifier = RelationshipClassifier(
            feature_dim=hidden_dim,
            num_classes=num_classes,
            hidden_dim=classifier_hidden_dim,
            dropout=classifier_dropout
        )

        # Contrastive loss for alignment
        self.contrastive_loss = ContrastiveLoss()

        print("\n" + "=" * 70)
        print("✅ Complete Baseline Model Initialized Successfully!")
        print("=" * 70)

        self._print_model_summary()

    def forward(self, images, captions, return_features=False):
        """
        Forward pass through the complete model

        Args:
            images: [batch_size, 3, 224, 224] - input image
            captions: List[str] - text captions (batch_size items)
            return_features: bool - whether to return intermediate features

        Returns:
            outputs: dict containing:
                - logits: [batch_size, num_classes]
                - probs: [batch_size, num_classes]
                - predictions: [batch_size]
                - similarity_matrix: [batch_size, batch_size] (for contrastive loss)
                - (optional) intermediate features if return_features=True
        """
        batch_size = images.size(0)

        # 1. Encode text
        text_features = self.text_encoder(captions)  # [batch, 256]

        # 2. Encode image
        image_features = self.image_encoder(images)  # [batch, 256]

        # 3. Multimodal alignment
        aligned_image, aligned_text, similarity_matrix = self.alignment(
            image_features, text_features
        )

        # 4. Fusion
        fused_features, fusion_weights = self.fusion(aligned_image, aligned_text)

        # 5. Classification
        logits, probs, predictions = self.classifier(fused_features)

        # Prepare outputs
        outputs = {
            'logits': logits,
            'probs': probs,
            'predictions': predictions,
            'similarity_matrix': similarity_matrix,
        }

        # Return intermediate features if requested (for analysis/visualization)
        if return_features:
            outputs.update({
                'text_features': text_features,
                'image_features': image_features,
                'aligned_image': aligned_image,
                'aligned_text': aligned_text,
                'fused_features': fused_features,
                'fusion_weights': fusion_weights,
            })

        return outputs

    def compute_loss(self, outputs, labels, alpha=0.1):
        """
        Compute total loss = Classification Loss + Contrastive Loss

        Args:
            outputs: dict from forward()
            labels: [batch_size] - ground truth labels
            alpha: weight for contrastive loss (default: 0.1)

        Returns:
            total_loss: scalar
            loss_dict: dict with individual losses for logging
        """
        # Classification loss (Cross Entropy)
        classification_loss = nn.functional.cross_entropy(
            outputs['logits'], labels
        )

        # Contrastive loss (for alignment)
        contrastive_loss = self.contrastive_loss(outputs['similarity_matrix'])

        # Total loss
        total_loss = classification_loss + alpha * contrastive_loss

        # Return individual losses for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'classification_loss': classification_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
        }

        return total_loss, loss_dict

    def _print_model_summary(self):
        """Print model architecture summary"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n📊 Model Architecture Summary:")
        print("-" * 70)

        # Component-wise parameter counts
        text_params = sum(p.numel() for p in self.text_encoder.parameters())
        text_trainable = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)

        image_params = sum(p.numel() for p in self.image_encoder.parameters())
        image_trainable = sum(p.numel() for p in self.image_encoder.parameters() if p.requires_grad)

        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())

        print(f"Text Encoder:        {text_params:>12,} params ({text_trainable:>12,} trainable)")
        print(f"Image Encoder:       {image_params:>12,} params ({image_trainable:>12,} trainable)")
        print(f"Alignment Module:    {0:>12,} params (no learnable params)")
        print(f"Fusion Module:       {fusion_params:>12,} params")
        print(f"Classifier:          {classifier_params:>12,} params")
        print("-" * 70)
        print(f"Total Parameters:    {total_params:>12,}")
        print(f"Trainable Parameters:{trainable_params:>12,}")
        print(f"Frozen Parameters:   {total_params - trainable_params:>12,}")
        print("-" * 70)

        # Memory estimate (rough)
        memory_mb = (total_params * 4) / (1024 ** 2)  # 4 bytes per float32
        print(f"Estimated Memory:    ~{memory_mb:.1f} MB")
        print()

    def get_config(self):
        """Return model configuration for saving"""
        return {
            'num_classes': self.num_classes,
            'hidden_dim': self.hidden_dim,
            'text_model_name': self.text_encoder.tokenizer.name_or_path,
        }


# Test code
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 Testing Complete Baseline Model")
    print("=" * 70)

    # Create model
    model = BaselineModel(
        num_classes=6,
        hidden_dim=256,
        pretrained_resnet=True
    )
    model.eval()

    # Test data
    print("\n📊 Creating test batch...")
    batch_size = 4

    # Sample image
    test_images = torch.randn(batch_size, 3, 224, 224)

    # Sample captions
    test_captions = [
        "Two friends playing basketball together",
        "A couple walking in the park",
        "Family members having dinner",
        "Colleagues working in an office"
    ]

    print(f"   Images shape: {test_images.shape}")
    print(f"   Captions: {len(test_captions)} samples")

    # Forward pass
    print(f"\n⚙️  Running forward pass...")
    with torch.no_grad():
        outputs = model(test_images, test_captions, return_features=True)

    print(f"\n✅ Forward pass successful!")
    print(f"\n📈 Output shapes:")
    print(f"   Logits: {outputs['logits'].shape}")  # [4, 6]
    print(f"   Probabilities: {outputs['probs'].shape}")  # [4, 6]
    print(f"   Predictions: {outputs['predictions'].shape}")  # [4]
    print(f"   Similarity Matrix: {outputs['similarity_matrix'].shape}")  # [4, 4]

    print(f"\n🔍 Intermediate features:")
    print(f"   Text features: {outputs['text_features'].shape}")  # [4, 256]
    print(f"   Image features: {outputs['image_features'].shape}")  # [4, 256]
    print(f"   Aligned image: {outputs['aligned_image'].shape}")  # [4, 256]
    print(f"   Aligned text: {outputs['aligned_text'].shape}")  # [4, 256]
    print(f"   Fused features: {outputs['fused_features'].shape}")  # [4, 256]
    print(f"   Fusion weights: {outputs['fusion_weights'].shape}")  # [4, 256]

    # Test loss computation
    print(f"\n📉 Testing loss computation...")
    test_labels = torch.randint(0, 6, (batch_size,))
    total_loss, loss_dict = model.compute_loss(outputs, test_labels, alpha=0.1)

    print(f"   Total loss: {loss_dict['total_loss']:.4f}")
    print(f"   Classification loss: {loss_dict['classification_loss']:.4f}")
    print(f"   Contrastive loss: {loss_dict['contrastive_loss']:.4f}")

    # Test predictions
    print(f"\n🎯 Sample predictions:")
    class_names = ["Friends", "Family", "Couple", "Professional", "Commercial", "No Relation"]
    for i in range(batch_size):
        pred = outputs['predictions'][i].item()
        conf = outputs['probs'][i, pred].item()
        print(f"   Sample {i + 1}: {class_names[pred]} ({conf:.1%} confidence)")

    # Test with different batch sizes
    print(f"\n🧪 Testing different batch sizes...")
    for bs in [1, 2, 8]:
        test_imgs = torch.randn(bs, 3, 224, 224)
        test_caps = ["Test caption"] * bs
        with torch.no_grad():
            out = model(test_imgs, test_caps)
        print(f"   Batch size {bs}: Output shape {out['logits'].shape} ✓")

    # Memory and speed test
    print(f"\n⚡ Speed test (10 iterations)...")
    import time

    model.eval()
    test_imgs = torch.randn(4, 3, 224, 224)
    test_caps = ["Test caption"] * 4

    # Warmup
    with torch.no_grad():
        _ = model(test_imgs, test_caps)

    # Timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_imgs, test_caps)
    end_time = time.time()

    avg_time = (end_time - start_time) / 10
    print(f"   Average inference time: {avg_time * 1000:.2f} ms/batch")
    print(f"   Throughput: {4 / avg_time:.1f} samples/sec")

    print("\n" + "=" * 70)
    print("✅ Complete Baseline Model Test PASSED!")
    print("=" * 70)

    print("\n🎉 Baseline Model Ready!")
    print("\n📋 Next Steps:")
    print("   1. Integrate with PISC dataset loader")
    print("   2. Create training loop")
    print("   3. Add evaluation metrics")
    print("   4. Start baseline training")