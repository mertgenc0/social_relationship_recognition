import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """
    L_total = L_CE(weighted) + α * L_contrastive
    """

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