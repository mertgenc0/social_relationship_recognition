"""
Multimodal Alignment Module for Baseline Model

Görüntü özellikleri  ile metin özelliklerini ortak bir anlamsal uzayda birleştirmek gerekir.
Bunun için Cosine Similarity (Kosinüs Benzerliği) matrisi kullanılır.

- Neden Uygulandı? Resimdeki piksellerle metindeki kelimelerin "aynı şeyi" ifade ettiğini modele öğretmek için
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

        # benzerlik skorlarını "keskinleştirmek" için kullanılır. doğru eşleşmeler daha güçlü odaklanamsını sağlar.
        self.temperature = temperature

        print(f"- Initializing Multimodal Alignment...")
        print(f"   Temperature: {temperature}")

    def compute_similarity_matrix(self, image_features, text_features):
        """
        Compute cosine similarity matrix between image and text features

        Args:
            image_features: [batch_size, feature_dim]
            text_features: [batch_size, feature_dim]

        Returns:
            similarity_matrix: [batch_size, batch_size]

        - görüntü ve metin vektörleri arasındaki açıyı ölçer. Eğer metin "el ele tutuşan çift" diyorsa ve görüntüde bu varsa,
         bu matristeki değer yükselir.
        """

        # Normalize features to unit length
        #Görüntü ve metin vektörlerinin uzunluklarını 1 birime eşitler.
        #Cosine Similarity (Kosinüs Benzerliği) hesaplamak için vektörlerin büyüklüklerinden kurtulup sadece aralarındaki açıya odaklanmamız gerekir.
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)


        # Compute cosine similarity: I @ T^T (matris çarpımı)
        # 4 resim ve 4 metin varsa, elimizde 4x4'lük bir tablo oluşur. Bu tablonun [i, j] hücresi, i. resmin j. metne ne kadar benzediğini söyler.
        similarity_matrix = torch.matmul(image_features, text_features.t())

        # Scale by temperature (Bölerek Softmax aşamasına hazırlar. Bu, gradyanların (öğrenme sinyallerinin) daha stabil olmasını sağlar.)
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

        ### Benzerlik skorlarını olasılıklara (0 ile 1 arası) dönüştürür. Satır toplamları 1 olur.

        # Softmax along text dimension (each image attends to all texts)
        # [batch, batch] -> [batch, batch]
        image_to_text_weights = F.softmax(similarity_matrix, dim=1)
        # Softmax along image dimension (each text attends to all image)
        # [batch, batch] -> [batch, batch]
        text_to_image_weights = F.softmax(similarity_matrix.t(), dim=1)

        ### Görüntü özelliklerini, ona benzeyen metin özelliklerinin ağırlıklı toplamıyla günceller.

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
    Encourages matching image-text pairs to have high similarity (modelin hata payını hesaplar.)
    - Doğru eşleşmelerin (doğru resim - doğru metin) birbirine yaklaşmasını, yanlışların uzaklaşmasını sağla
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

        #Eğer 4 örneğimiz varsa, etiketler [0, 1, 2, 3] olur. Bu, "0. resim 0. metinle eşleşmeli",
        #"1. resim 1. metinle eşleşmeli" demektir. Matrisin köşegeni (diagonal) bizim hedefimizdir.

        # Ground truth: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=similarity_matrix.device)

        ### Hem "resme göre doğru metni bulma" hem de "metne göre doğru resmi bulma" hatalarını hesaplayıp ortalamasını alır.

        # Image-to-text loss
        loss_i2t = F.cross_entropy(similarity_matrix, labels)
        # Text-to-image loss
        loss_t2i = F.cross_entropy(similarity_matrix.t(), labels)
        # Average both directions
        loss = (loss_i2t + loss_t2i) / 2

        return loss
class IterativeAlignment(nn.Module):
    def __init__(self, feature_dim=256, K=3):
        super(IterativeAlignment, self).__init__()
        self.K = K # Varsayılan 3 iterasyon [cite: 194, 226]
        self.text_to_vis = nn.ModuleList([nn.MultiheadAttention(feature_dim, 8) for _ in range(K)])
        self.vis_to_text = nn.ModuleList([nn.MultiheadAttention(feature_dim, 8) for _ in range(K)])
        self.ln_v = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(K)])
        self.ln_t = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(K)])

    def forward(self, v, t):
        v, t = v.unsqueeze(0), t.unsqueeze(0) # [1, B, D]
        for k in range(self.K):
            # Text attends to Visual [cite: 197-202]
            t_att, _ = self.text_to_vis[k](t, v, v)
            t = self.ln_t[k](t + t_att)
            # Visual attends to Text [cite: 208-210]
            v_att, _ = self.vis_to_text[k](v, t, t)
            v = self.ln_v[k](v + v_att)
        return v.squeeze(0), t.squeeze(0)

# Test Kodu
if __name__ == "__main__":
    print("-----Çok Modlu Hizalama Modülü Test Ediliyor-----")

    # Hizalama modülünü oluştur
    alignment = MultimodalAlignment(temperature=0.07)
    alignment.eval()

    # Kayıp fonksiyonunu oluştur
    contrastive_loss = ContrastiveLoss()

    # Örnek özelliklerle test et
    print(f"\n-  Test özellikleri oluşturuluyor...")
    batch_size = 4
    feature_dim = 256

    # Kodlayıcılardan (encoders) gelen sahte görüntü ve metin vektörleri
    image_features = torch.randn(batch_size, feature_dim)
    text_features = torch.randn(batch_size, feature_dim)

    print(f"   Görüntü özellik boyutu: {image_features.shape}")
    print(f"   Metin özellik boyutu: {text_features.shape}")

    # İleri besleme (Forward pass)
    print(f"\n-   Hizalama işlemi çalıştırılıyor...")
    with torch.no_grad():
        aligned_image, aligned_text, similarity_matrix = alignment(
            image_features, text_features
        )

    print(f"\n-  Hizalama başarılı!")
    print(f"   Hizalanmış görüntü boyutu: {aligned_image.shape}")
    print(f"   Hizalanmış metin boyutu: {aligned_text.shape}")
    print(f"   Benzerlik matrisi boyutu: {similarity_matrix.shape}")

    # Benzerlik matrisini incele
    print(f"\n-  Benzerlik Matrisi Analizi:")
    print(f"   Matris içeriği:\n{similarity_matrix}")
    print(f"   Köşegen (Eşleşen çiftler): {similarity_matrix.diag()}")
    print(f"   Ortalama benzerlik: {similarity_matrix.mean():.3f}")
    print(f"   Köşegen ortalaması: {similarity_matrix.diag().mean():.3f}")

    # Kayıp fonksiyonunu test et
    print(f"\n-  Karşılaştırmalı Kayıp (Loss) Test Ediliyor...")
    loss = contrastive_loss(similarity_matrix)
    print(f"   Hesaplanan Kayıp: {loss.item():.4f}")

    # Mükemmel hizalanmış özelliklerle test et
    print(f"\n-  Mükemmel hizalanmış (aynı) özelliklerle test ediliyor...")
    perfect_features = torch.randn(batch_size, feature_dim)
    with torch.no_grad():
        _, _, perfect_sim = alignment(perfect_features, perfect_features)
        perfect_loss = contrastive_loss(perfect_sim)

    print(f"   Mükemmel benzerlik matrisi:\n{perfect_sim}")
    print(f"   Mükemmel hizalama kaybı: {perfect_loss.item():.4f}")
    print(f"   (Düşük kayıp = Daha iyi hizalama)")

    # Yanlış hizalanmış özelliklerle test et
    print(f"\n-  Yanlış hizalanmış özelliklerle test ediliyor...")
    misaligned_image = torch.randn(batch_size, feature_dim)
    misaligned_text = torch.randn(batch_size, feature_dim)
    with torch.no_grad():
        _, _, misaligned_sim = alignment(misaligned_image, misaligned_text)
        misaligned_loss = contrastive_loss(misaligned_sim)

    print(f"   Yanlış hizalama kaybı: {misaligned_loss.item():.4f}")
    print(f"   (Yüksek kayıp = Kötü hizalama)")

    # Model İstatistikleri
    param_count = sum(p.numel() for p in alignment.parameters())
    print(f"\n- Model İstatistikleri:")
    print(f"   Toplam parametre: {param_count}")
    print(f"   (Bu modül eğitilebilir parametre içermez, sadece matematiksel işlem yapar.)")


    print("----Çok Modlu Hizalama Testi TAMAMLANDI!-----")

