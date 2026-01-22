"""
Complete Baseline Model for Social Relationship Recognition
Combines all components: Text Encoder + Image Encoder + Alignment + Fusion + Classifier
"""

import torch
import torch.nn as nn
from .image_encoder import ResNetWithAttention, FPNImageEncoder
from .alignment import MultimodalAlignment, ContrastiveLoss, IterativeAlignment
from .fusion import AdaptiveFusion, UncertaintyFusion
from .classifier import RelationshipClassifier
from .text_encoder import LLMTextEncoder

class BaselineModel(nn.Module):
    def __init__(
            self,
            num_classes=6,
            use_enhanced=False, # İnovasyonları açıp kapatan anahtar
            hidden_dim=256,
            text_model_name='bert-base-uncased',
            pretrained_resnet=True,
            alignment_temperature=0.07,
            fusion_hidden_dim=128,
            classifier_hidden_dim=128,
            classifier_dropout=0.3,
            class_weights=None
    ):
        super(BaselineModel, self).__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.use_enhanced = use_enhanced
        if class_weights is not None:
            self.register_buffer('weights', class_weights)
        else:
            self.weights = None


        print("=" * 70)
        mode_name = "ENHANCED (Innovation Mode)" if use_enhanced else "BASELINE Mode"
        print(f"🚀 Initializing Model in {mode_name}")
        print("=" * 70)

        # 1. Text Encoder (Baseline)
        self.text_encoder = LLMTextEncoder(model_name=text_model_name, hidden_dim=hidden_dim)

        # 2. Image Encoder (Baseline + Innovation 1)
        self.image_encoder = ResNetWithAttention(hidden_dim=hidden_dim, pretrained=pretrained_resnet)
        if self.use_enhanced:
            self.fpn_enhancer = FPNImageEncoder(hidden_dim=hidden_dim)

        # 3. Alignment (Baseline + Innovation 2)
        self.alignment = MultimodalAlignment(temperature=alignment_temperature)
        if self.use_enhanced:
            self.iterative_refiner = IterativeAlignment(feature_dim=hidden_dim, K=3)

        # 4. Fusion (Baseline + Innovation 3)
        self.fusion = AdaptiveFusion(feature_dim=hidden_dim, hidden_dim=fusion_hidden_dim)
        if self.use_enhanced:
            self.uncertainty_fusion = UncertaintyFusion(feature_dim=hidden_dim)

        # 5. Classifier
        self.classifier = RelationshipClassifier(
            feature_dim=hidden_dim,
            num_classes=num_classes,
            hidden_dim=classifier_hidden_dim,
            dropout=classifier_dropout
        )

        self.contrastive_loss = ContrastiveLoss()
        self._print_model_summary()

    def forward(self, images, captions, return_features=False):
        # 1. Baseline Özellik Çıkarımı
        text_features = self.text_encoder(captions)
        image_features = self.image_encoder(images)

        # İnovasyon 1: FPN Enhancer (Eğer aktifse baseline üzerine biner)
        if self.use_enhanced:
            image_features = self.fpn_enhancer(images)

        # 2. Alignment (Baseline hizalama yapılır)
        aligned_image, aligned_text, similarity_matrix = self.alignment(image_features, text_features)

        # İnovasyon 2: Iterative Refinement
        if self.use_enhanced:
            aligned_image, aligned_text = self.iterative_refiner(aligned_image, aligned_text)
            similarity_matrix = torch.matmul(aligned_image, aligned_text.t())

        # 3. Fusion
        fused_features, fusion_weights = self.fusion(aligned_image, aligned_text)

        # İnovasyon 3: Uncertainty-Aware Fusion
        uncertainty = None
        if self.use_enhanced:
            fused_features, uncertainty = self.uncertainty_fusion(aligned_image, aligned_text)

        # 4. Classification
        logits, probs, predictions = self.classifier(fused_features)

        outputs = {
            'logits': logits,
            'probs': probs,
            'predictions': predictions,
            'similarity_matrix': similarity_matrix,
            'uncertainty': uncertainty
        }

        if return_features:
            outputs.update({
                'text_features': text_features,
                'image_features': image_features,
                'aligned_image': aligned_image,
                'aligned_text': aligned_text,
                'fused_features': fused_features
            })

        return outputs

    def compute_loss(self, outputs, labels, alpha=0.1):
        """
        Makaleye (Wang ve ark.) %100 sadık kayıp fonksiyonu.
        L_total = 0.3 * L_CE(weighted) + alpha * L_cont
        """
        # Sınıflandırma Kaybı (Ağırlıklı Cross Entropy)
        # Sınıf ağırlıkları burada devreye girerek Professional baskınlığını kırar.
        classification_loss = nn.functional.cross_entropy(
            outputs['logits'],
            labels,
            weight=self.weights
        )

        # Hizalama Kaybı (Contrastive Loss)
        contrastive_loss = self.contrastive_loss(outputs['similarity_matrix'])

        # Makaledeki Modality Balance Factor (0.3) katsayısı
        total_loss = (0.3 * classification_loss) + (alpha * contrastive_loss)

        loss_dict = {
            'total': total_loss.item(),
            'classification': classification_loss.item(),
            'contrastive': contrastive_loss.item()
        }

        # İnovasyon 3: Belirsizlik Düzenlemesi (Enhanced Mod Açıksa)
        if self.use_enhanced and outputs.get('uncertainty') is not None:
            sig_v, sig_t = outputs['uncertainty']
            unc_loss = torch.mean(torch.abs(sig_v - 0.5) + torch.abs(sig_t - 0.5))
            total_loss += 0.05 * unc_loss
            loss_dict['total'] = total_loss.item()
            loss_dict['uncertainty_loss'] = unc_loss.item()

        return total_loss, loss_dict

    def get_config(self):
        """Modeli kaydetmek ve konfigürasyonu takip etmek için"""
        return {
            'num_classes': self.num_classes,
            'hidden_dim': self.hidden_dim,
            'use_enhanced': self.use_enhanced,
            'text_model_name': self.text_encoder.tokenizer.name_or_path if hasattr(self.text_encoder, 'tokenizer') else 'bert-base-uncased'
        }

    def _print_model_summary(self):
        """Modelin parametre sayısını ve modül özetini yazdırır"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n📊 Model Mimarisi Özeti:")
        print("-" * 50)
        print(f"Mod: {'GELİŞMİŞ (İnovasyonlar Aktif)' if self.use_enhanced else 'BASELINE (Sadece Temel Yapı)'}")
        print(f"Toplam Parametre: {total_params:,}")
        print(f"Eğitilebilir Parametre: {trainable_params:,}")
        print(f"Dondurulmuş Parametre: {total_params - trainable_params:,}")
        print("-" * 50)


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