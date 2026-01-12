"""
Text Encoder Module for Baseline Model:

Metinlerden olay yapılarını (duygular, sahneler, ilişkiler) çıkarmak için bir LLM (Büyük Dil Modeli) kullanılması.
Ardından bu yapılandırılmış bilgiyi bir CNN (Evrişimli Sinir Ağı) ile işleyerek metin vektörlerine dönüştürülür.

LLM Model: bert-base-uncased

Neden Uygulandı? Sadece düz metni değil, metindeki "mutlu", "park", "çift" gibi anahtar kelimelerin birbirleriyle olan
mantıksal bağını vektörel bir düzleme dökmek için
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class LLMTextEncoder(nn.Module):

    # model ve modelin sonunda üreteceği vektörün uzunluğu
    def __init__(self, model_name='bert-base-uncased', hidden_dim=256):
        super(LLMTextEncoder, self).__init__()

        print(f"Initializing Text Encoder with {model_name}.")

        # LLM (BERT)
        # Metni, bilgisayarın anlayacağı sayılara (token) böler
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # önceden eğitilmiş ana BERT modeli
        self.llm = AutoModel.from_pretrained(model_name)

        # modelinin ağırlıklarını sabitler eğitilmesini engeller.Bellek tasarrufu sağlar ve sadece cnn katmanalrı öğrenemsini sağlar
        for param in self.llm.parameters():
            param.requires_grad = False

        llm_dim = self.llm.config.hidden_size  # 768 for BERT-base

        # Text CNN (from baseline paper)
        # Kernel size = 3, stride = 1 (as per Algorithm 1)
        self.text_cnn = nn.Sequential(
            nn.Conv1d(llm_dim, hidden_dim * 2, kernel_size=3, padding=1),  # Metindeki keliem gruplarını tarar
            nn.ReLU(),  # non-linear özellik katar. (negatif değerleri sıfılayarak)
            nn.MaxPool1d(kernel_size=2),  # en önemli özellikleri seçip veri boyutunu yarıya indirir
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # çıktıyı sabit bir boyuta indrger
        )

        # Final projection
        # Özellikleri son bir kez işleyerek temsil vektörünü oluşturur Fc
        self.fc = nn.Linear(hidden_dim, hidden_dim)

        print(f"   Text Encoder initialized")
        print(f"   LLM dimension: {llm_dim}")
        print(f"   Output dimension: {hidden_dim}")

    def forward(self, captions):
        # Tokenize captions
        # Metinleri alır, hepsini aynı uzunluğa (77 token) getirir ve PyTorch Tensörüne dönüştürür.
        device = next(self.llm.parameters()).device

        tokens = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=77,  # Standard for CLIP-like models
            return_tensors='pt'
        ).to(device)

        # LLM encoding (frozen, no gradients)

        # modelini çalıştırır. last_hidden_state, her kelime için 768 boyutunda bir vektör üretir. torch.no_grad() gradyan hesaplamayarak hızı artırır.
        with torch.no_grad():
            outputs = self.llm(**tokens)
            # Get last hidden state: [batch, seq_len, 768]
            text_embeddings = outputs.last_hidden_state

        # Transpose for CNN: [batch, 768, seq_len] şekline getirerek. format değişikliği yapar
        text_embeddings = text_embeddings.permute(0, 2, 1)

        # CNN feature extraction
        features = self.text_cnn(text_embeddings)  # [batch, hidden_dim, 1]
        features = features.squeeze(-1)  # [batch, hidden_dim]

        # Final projection
        text_features = self.fc(features)

        return text_features

    def get_output_dim(self):
        """Return output feature dimension"""
        return self.fc.out_features


# Test code
if __name__ == "__main__":
    print("-----Testing Text Encoder Module-----")

    # Create model
    model = LLMTextEncoder(hidden_dim=256)
    model.eval()

    # Test with sample captions
    captions = [
        "Two friends are playing basketball on the court",
        "A couple walking hand in hand in the park",
        "Family members having dinner together",
        "Colleagues working in an office"
    ]

    print(f"\n- Test caption ({len(captions)} sampels:")
    for i, cap in enumerate(captions):
        print(f"   {i + 1}. {cap}")

    # Forward pass
    print(f"\n- Running forward pass.")
    with torch.no_grad():
        features = model(captions)

    print(f"\n- Forward pass successful!")
    print(f"   Input: {len(captions)} captions")
    print(f"   Output shape: {features.shape}")  # Should be [4, 256]
    print(f"   Output dtype: {features.dtype}")
    print(f"   Output range: [{features.min():.3f}, {features.max():.3f}]")

    # Test with single caption
    print(f"\n- Testing single caption...")
    single_caption = ["A young couple smiling together"]
    with torch.no_grad():
        single_feature = model(single_caption)

    print(f"Single output shape: {single_feature.shape}")  # Should be [1, 256]

    # Memory usage
    # Toplam parametre sayısını ve bizim eğitebileceğimiz (BERT hariç) parametre sayısını hesaplar.
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n- Model Statistics:")
    print(f"   Total parameters: {param_count:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {param_count - trainable_params:,}")

    print("-------Text Encoder Test PASSED--------")

# Output

"""
-----Testing Text Encoder Module-----
Initializing Text Encoder with bert-base-uncased.
   Text Encoder initialized
   LLM dimension: 768
   Output dimension: 256

- Test caption (4 sampels:
   1. Two friends are playing basketball on the court
   2. A couple walking hand in hand in the park
   3. Family members having dinner together
   4. Colleagues working in an office

- Running forward pass.

- Forward pass successful!
   Input: 4 captions
   Output shape: torch.Size([4, 256])
   Output dtype: torch.float32
   Output range: [-0.164, 0.184]

- Testing single caption...
Single output shape: torch.Size([1, 256])

- Model Statistics:
   Total parameters: 111,121,664
   Trainable parameters: 1,639,424
   Frozen parameters: 109,482,240
-------Text Encoder Test PASSED--------
"""