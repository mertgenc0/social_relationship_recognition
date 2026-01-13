"""
Classifier Module for Baseline Model
Final classification layer for relationship prediction
"""

import torch
import torch.nn as nn


class RelationshipClassifier(nn.Module):
    """
    Multi-layer classifier for relationship prediction

    Pipeline:
    1. Fused features → [batch, feature_dim]
    2. FC layers with dropout
    3. Output logits → [batch, num_classes]
    4. Softmax for probabilities

    - Birleştirilmiş vektör bir Softmax fonksiyonuna sokularak ilişkinin türü (Arkadaş, Aile, Sevgili vb.) tahmin edilir

    - Neden Uygulandı? Modelin elindeki sayısal veriyi, insanların anlayabileceği "Yüzde 90 ihtimalle Sevgili" gibi bir etikete dönüştürmek için.
    """

    def __init__(self, feature_dim=256, num_classes=6, hidden_dim=128, dropout=0.3):
        super(RelationshipClassifier, self).__init__()

        print(f"🔧 Initializing Relationship Classifier...")

        self.num_classes = num_classes

        # Multi-layer classifier
        self.classifier = nn.Sequential(
            # First layer
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Second layer
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Output layer
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

        print(f"- Classifier initialized")
        print(f"   Input dimension: {feature_dim}")
        print(f"   Hidden dimension: {hidden_dim}")
        print(f"   Number of classes: {num_classes}")
        print(f"   Dropout rate: {dropout}")

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, fused_features):
        """
        Classify relationship from fused features

        Args:
            fused_features: [batch_size, feature_dim]

        Returns:
            logits: [batch_size, num_classes] - raw scores
            probs: [batch_size, num_classes] - probabilities (softmax)
            predictions: [batch_size] - predicted class indices
        """
        # Get logits
        logits = self.classifier(fused_features)  # [batch, num_classes]

        # Compute probabilities
        probs = torch.softmax(logits, dim=-1)  # [batch, num_classes]

        # Get predictions
        predictions = torch.argmax(probs, dim=-1)  # [batch]

        return logits, probs, predictions


class SimpleClassifier(nn.Module):
    """
    Simple single-layer classifier for ablation studies
    """

    def __init__(self, feature_dim=256, num_classes=6):
        super(SimpleClassifier, self).__init__()

        self.num_classes = num_classes
        self.classifier = nn.Linear(feature_dim, num_classes)

        # Initialize
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, fused_features):
        """Simple linear classification"""
        logits = self.classifier(fused_features)
        probs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1)
        return logits, probs, predictions


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing Classifier Module")
    print("=" * 60)

    # Define relationship classes (PISC dataset)
    CLASS_NAMES = [
        "Friends",
        "Family",
        "Couple",
        "Professional",
        "Commercial",
        "No Relation"
    ]

    # Create classifiers
    print("\n📊 Testing Multi-layer Classifier...")
    classifier = RelationshipClassifier(
        feature_dim=256,
        num_classes=6,
        hidden_dim=128,
        dropout=0.3
    )
    classifier.eval()

    print("\n-----Testing Simple Classifier-----")
    simple_classifier = SimpleClassifier(feature_dim=256, num_classes=6)
    simple_classifier.eval()

    # Test with sample fused features
    print(f"\n📊 Creating test fused features...")
    batch_size = 4
    feature_dim = 256

    fused_features = torch.randn(batch_size, feature_dim)
    print(f"   Fused features shape: {fused_features.shape}")

    # Test Multi-layer Classifier
    print(f"\n⚙️  Running multi-layer classifier...")
    with torch.no_grad():
        logits, probs, predictions = classifier(fused_features)

    print(f"\n✅ Multi-layer classifier successful!")
    print(f"   Logits shape: {logits.shape}")  # [4, 6]
    print(f"   Probabilities shape: {probs.shape}")  # [4, 6]
    print(f"   Predictions shape: {predictions.shape}")  # [4]

    print(f"\n🔍 Sample predictions:")
    for i in range(batch_size):
        pred_class = predictions[i].item()
        pred_prob = probs[i, pred_class].item()
        print(f"   Sample {i + 1}: {CLASS_NAMES[pred_class]} ({pred_prob:.1%} confidence)")
        # Show top-2 predictions
        top2_probs, top2_indices = torch.topk(probs[i], k=2)
        print(f"      Top-2: {CLASS_NAMES[top2_indices[0].item()]} ({top2_probs[0].item():.1%}), "
              f"{CLASS_NAMES[top2_indices[1].item()]} ({top2_probs[1].item():.1%})")

    # Verify probabilities sum to 1
    print(f"\n🔍 Probability verification:")
    prob_sums = probs.sum(dim=-1)
    print(f"   Probability sums: {prob_sums}")
    print(f"   All close to 1.0: {torch.allclose(prob_sums, torch.ones_like(prob_sums))}")

    # Test Simple Classifier
    print(f"\n⚙️  Running simple classifier...")
    with torch.no_grad():
        simple_logits, simple_probs, simple_preds = simple_classifier(fused_features)

    print(f"\n✅ Simple classifier successful!")
    print(f"   Logits shape: {simple_logits.shape}")
    print(f"   Predictions: {simple_preds}")

    # Test with single sample
    print(f"\n🔍 Testing single sample...")
    single_feature = torch.randn(1, feature_dim)
    with torch.no_grad():
        single_logits, single_probs, single_pred = classifier(single_feature)

    print(f"   Single prediction: {CLASS_NAMES[single_pred.item()]}")
    print(f"   Confidence: {single_probs[0, single_pred].item():.1%}")

    # Test confidence analysis
    print(f"\n📊 Confidence Analysis:")
    with torch.no_grad():
        test_features = torch.randn(100, feature_dim)
        _, test_probs, test_preds = classifier(test_features)

    # Get confidence for predictions
    confidences = test_probs.gather(1, test_preds.unsqueeze(1)).squeeze()
    print(f"   Mean confidence: {confidences.mean():.1%}")
    print(f"   Median confidence: {confidences.median():.1%}")
    print(f"   Min confidence: {confidences.min():.1%}")
    print(f"   Max confidence: {confidences.max():.1%}")

    # Class distribution
    print(f"\n📊 Predicted Class Distribution (100 samples):")
    for i, class_name in enumerate(CLASS_NAMES):
        count = (test_preds == i).sum().item()
        print(f"   {class_name}: {count}/100 ({count}%)")

    # Memory usage
    classifier_params = sum(p.numel() for p in classifier.parameters())
    classifier_trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)

    simple_params = sum(p.numel() for p in simple_classifier.parameters())
    simple_trainable = sum(p.numel() for p in simple_classifier.parameters() if p.requires_grad)

    print(f"\n📊 Model Statistics:")
    print(f"   Multi-layer Classifier:")
    print(f"     Total parameters: {classifier_params:,}")
    print(f"     Trainable parameters: {classifier_trainable:,}")
    print(f"   Simple Classifier:")
    print(f"     Total parameters: {simple_params:,}")
    print(f"     Trainable parameters: {simple_trainable:,}")

    print("\n" + "=" * 60)
    print("✅ Classifier Module Test PASSED!")
    print("=" * 60)

    print("\n💡 Practical Interpretation:")
    print("   - Multi-layer classifier provides better feature transformation")
    print("   - Dropout prevents overfitting")
    print("   - LayerNorm stabilizes training")
    print("   - Confidence scores indicate prediction reliability")
    print("   - Simple classifier is faster but less expressive")