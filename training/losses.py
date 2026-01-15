import torch
import torch.nn as nn
import torch.nn.functional as F


# training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self, num_classes=6, alpha=0.1, label_smoothing=0.0, weight=None, use_enhanced=False):
        super(CombinedLoss, self).__init__()
        self.use_enhanced = use_enhanced
        self.num_classes = num_classes
        self.alpha = alpha  # Kontrastif kayıp ağırlığı (PDF'de lambda_1/alpha)

        # Sınıf ağırlıklı Cross Entropy
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            weight=weight
        )

        print(f"🔧 Initializing Combined Loss...")
        print(f"   Mode: {'ENHANCED' if use_enhanced else 'BASELINE'}")
        print(f"   Contrastive weight (α): {alpha}")

    def classification_loss(self, logits, labels):
        return self.ce_loss(logits, labels)

    def contrastive_loss(self, similarity_matrix):
        batch_size = similarity_matrix.size(0)
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        loss_i2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2i = F.cross_entropy(similarity_matrix.t(), labels)
        return (loss_i2t + loss_t2i) / 2

    def forward(self, outputs, labels):
        # 1. Sınıflandırma Kaybı
        cls_loss = self.classification_loss(outputs['logits'], labels)

        # 2. Kontrastif (Hizalama) Kaybı
        cont_loss = self.contrastive_loss(outputs['similarity_matrix'])

        # MAKALEYE SADIK: Modality Balance Factor (lambda = 0.3)
        total_loss = (0.3 * cls_loss) + (self.alpha * cont_loss)

        loss_dict = {
            'total': total_loss.item(),
            'classification': cls_loss.item(),
            'contrastive': cont_loss.item(),
        }

        # İNOVASYON 3: Uncertainty Regularization Loss (Equation 16)
        # Belirsizlik değerlerinin (sigma) 0.5 civarında stabilize olmasını sağlar.
        if self.use_enhanced and outputs.get('uncertainty') is not None:
            sig_v, sig_t = outputs['uncertainty']
            # Denklem 16: L_uncertainty = |sigma_v - 0.5| + |sigma_t - 0.5|
            unc_loss = torch.mean(torch.abs(sig_v - 0.5) + torch.abs(sig_t - 0.5))

            # Toplama ekle (lambda_2 = 0.05)
            total_loss += 0.05 * unc_loss
            loss_dict['uncertainty_reg'] = unc_loss.item()

        return total_loss, loss_dict

"""
class CombinedLoss(nn.Module):
    

    def __init__(self, num_classes=6, alpha=0.1, label_smoothing=0.0, weight=None):
        super(CombinedLoss, self).__init__()

        self.num_classes = num_classes
        self.alpha = alpha

        self.lambda_factor = 0.3

        # Sınıf ağırlıkları (weight) buraya entegre edildi
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            weight=weight
        )

        print(f"🔧 Initializing Combined Loss...")
        print(f"   Number of classes: {num_classes}")
        print(f"   Contrastive weight (α): {alpha}")
        print(f"   Class Weights Active: {weight is not None}")

    def classification_loss(self, logits, labels):
        return self.ce_loss(logits, labels)

    def contrastive_loss(self, similarity_matrix):
        batch_size = similarity_matrix.size(0)
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        loss_i2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2i = F.cross_entropy(similarity_matrix.t(), labels)
        return (loss_i2t + loss_t2i) / 2

    def forward(self, outputs, labels):
        cls_loss = self.classification_loss(outputs['logits'], labels)
        cont_loss = self.contrastive_loss(outputs['similarity_matrix'])

        # MAKALEYE SADIK DÜZELTME: Modality Balance Factor (lambda = 0.3)
        # Bu faktör, görsel ve metinsel modlar arasındaki gradyan dengesini sağlar.
        total_loss = (0.3 * cls_loss) + (self.alpha * cont_loss)

        loss_dict = {
            'total': total_loss.item(),
            'classification': cls_loss.item(),
            'contrastive': cont_loss.item(),
        }
        return total_loss, loss_dict
"""