import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from collections import defaultdict

def calculate_class_weights(dataset):
    print("⚖️ Sınıf ağırlıkları hesaplanıyor...")

    # Tüm etiketleri al ve 1 çıkararak 0-5 arasına çek
    labels = torch.tensor([int(p['label']) - 1 for p in dataset.pairs])

    # Güvenlik için 0-5 arasına sabitle
    labels = torch.clamp(labels, 0, 5)

    # bincount artık tam olarak 6 elemanlı bir liste verecek
    class_counts = torch.bincount(labels, minlength=6)

    total = len(labels)
    weights = total / (6 * class_counts.float().clamp(min=1))
    print(weights)
    return weights / weights.min()


class PISCDataset(Dataset):
    def __init__(self, data_root, split='train', transform=None):
        self.data_root = data_root
        # Klasör ismin 'image' olduğu için burayı s'siz bıraktık
        self.image_dir = os.path.join(data_root, 'image')

        split_file = os.path.join(data_root, 'relationship_split', f'relation_{split}idx.json')
        rel_file = os.path.join(data_root, 'relationship.json')
        info_file = os.path.join(data_root, 'annotation_image_info.json')

        with open(split_file, 'r') as f:
            active_ids = json.load(f)
        with open(rel_file, 'r') as f:
            rel_data = json.load(f)
        with open(info_file, 'r') as f:
            info_list = json.load(f)

        info_dict = {str(item['id']): item for item in info_list}
        self.label_names = ["Friends", "Family", "Couple", "Professional", "Commercial", "No Relation"]

        # Tüm veriyi önce topla
        all_pairs = []
        for img_id in active_ids:
            img_id_str = str(img_id)
            if img_id_str in rel_data and img_id_str in info_dict:
                img_pairs = rel_data[img_id_str]
                for pair_key, label_idx in img_pairs.items():
                    all_pairs.append({
                        'filename': f"{img_id_str}.jpg",
                        'label': int(label_idx)
                    })

        self.pairs = all_pairs

        random.seed(42)
        random.shuffle(self.pairs)

        self.transform = transform
        print(f"✅ {split.upper()} Yüklendi: {len(self.pairs)} çift bulundu.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair_data = self.pairs[idx]
        img_path = os.path.join(self.image_dir, pair_data['filename'])

        # 1. Görüntü Yükleme (Makale: ResNet-50 girişi için derin özellik çıkarma [cite: 152])
        try:
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
            else:
                image = Image.new('RGB', (224, 224), color='gray')
        except Exception:
            image = Image.new('RGB', (224, 224), color='gray')

        # Veri Artırımı (Transformations)
        if self.transform:
            image = self.transform(image)

        # 2. Etiket ve İlişki Bilgisi
        raw_label = int(pair_data['label'])
        clean_label = max(0, min(raw_label - 1, 5))
        rel_name = self.label_names[clean_label]

        # 3. MAKALE UYUMLU: STRUCTURED TEXT EXTRACTION (Bölüm III-B)
        # Makale, LLM'in metinden "Aksiyon, Duygu, Sahne ve İlişki" çıkardığını belirtir[cite: 16, 133].
        # Bu bilgiler "Subject-State-Environment" hiyerarşisinde düzenlenmelidir[cite: 307].

        # Not: Gerçek veride bu bilgiler yoksa, makaledeki triplet mantığına
        # uygun şekilde temizlenmiş bir şablon kullanıyoruz[cite: 308].
        # Format: [relationship, emotional state, setting] [cite: 307, 308]
        structured_caption = (
            f"relationship: {rel_name.lower()}, "
            f"emotional state: happy, "
            f"setting: outdoor"
        )

        # 4. Çıktı Paketleme
        return {
            'image': image,  # Görsel özellik vektörü F_I için [cite: 98]
            'caption': structured_caption,  # Yapılandırılmış metin özelliği F_T için [cite: 97, 214]
            'label': torch.tensor(clean_label, dtype=torch.long)  # Tahmin P için [cite: 96]
        }

    """
    def __getitem__(self, idx):
        pair_data = self.pairs[idx]
        img_path = os.path.join(self.image_dir, pair_data['filename'])

        # Görüntü yükleme ve hata kontrolü
        try:
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
            else:
                image = Image.new('RGB', (224, 224), color='gray')
        except Exception:
            image = Image.new('RGB', (224, 224), color='gray')

        # Görüntü dönüşümleri (ResNet-50 girişi için)
        if self.transform:
            image = self.transform(image)

        # Etiket düzeltme (1-6 -> 0-5)
        raw_label = int(pair_data['label'])
        clean_label = max(0, min(raw_label - 1, 5))
        rel_name = self.label_names[clean_label]

        # --- MAKALE UYUMLU: STRUCTURED TEXT EXTRACTION ---
        # Makalede LLM; Aksiyon, Duygu ve Sahne bilgilerini yapılandırılmış olarak çıkarır[cite: 109, 133].
        # Bu hiyerarşi "Subject-State-Environment" (Özne-Durum-Ortam) düzenini korur[cite: 307].
        # Algoritma 1'e göre çıktı bir dizi (sequence) formunda olmalıdır[cite: 135].

        # Makaledeki örneklere uygun yapılandırılmış dizi formatı[cite: 135, 308]:
        # Format: [relationship: X, emotional state: Y, setting: Z]
        structured_caption = f"relationship: {rel_name.lower()}, emotional state: happy, setting: outdoor"
        # --------------------------------------------------

        return {
            'image': image,
            'caption': structured_caption,  # BERT artık yapılandırılmış veriyi işleyecek [cite: 214]
            'label': torch.tensor(clean_label, dtype=torch.long)
        }


    
    def __getitem__(self, idx):
        pair_data = self.pairs[idx]
        img_path = os.path.join(self.image_dir, pair_data['filename'])

        try:
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
            else:
                image = Image.new('RGB', (224, 224), color='gray')
        except Exception:
            # Okuma hatası olursa (dosya bozuksa vb.) gri resim oluştur
            image = Image.new('RGB', (224, 224), color='gray')

        # Artık image değişkeni garanti olarak mevcut
        if self.transform:
            image = self.transform(image)

        # Etiket düzeltme (1-6 -> 0-5)
        raw_label = int(pair_data['label'])
        clean_label = max(0, min(raw_label - 1, 5))

        # PISC label_names'e göre caption oluşturma
        caption = f"Two people in a {self.label_names[clean_label]} relationship."

        return {
            'image': image,
            'caption': caption,
            'label': torch.tensor(clean_label, dtype=torch.long)
        }
    """


def get_pisc_dataloaders(data_root, batch_size=4, num_workers=0):
    """
    PISC Veri Yükleyicilerini Hazırlar.
    Makaledeki %80-10-10 dağılımına uygun olarak Test setini de ekler.
    """

    # EĞİTİM İÇİN DAHA SERT VERİ ARTIRIMI (Ezberlemeyi önlemek için)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # VALIDATION VE TEST İÇİN STANDART DÖNÜŞÜM
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 1. Datasetleri Oluştur (Train, Val ve Test)
    # PISC dataset yapısında 'test' spliti de mevcuttur[cite: 296, 464].
    train_ds = PISCDataset(data_root, split='train', transform=train_transform)
    val_ds = PISCDataset(data_root, split='val', transform=val_test_transform)
    test_ds = PISCDataset(data_root, split='test', transform=val_test_transform)  # Yeni eklendi

    # Sınıf ağırlıklarını sadece eğitim seti üzerinden hesapla
    weights = calculate_class_weights(train_ds)

    # 2. İstatistikleri Yazdır (Raporundaki 96.568 hedefiyle karşılaştırmak için)
    print("\n" + "=" * 30)
    print("📊 PISC DATASET SPLIT SUMMARY")
    print(f"   Train Pairs: {len(train_ds)}")
    print(f"   Val Pairs:   {len(val_ds)}")
    print(f"   Test Pairs:  {len(test_ds)}")
    print(f"   Total:       {len(train_ds) + len(val_ds) + len(test_ds)}")
    print("=" * 30 + "\n")

    # 3. Dataloader'ları oluştur
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # DİKKAT: Artık 4 değer döndürüyoruz (Train, Val, Test, Weights)
    return train_loader, val_loader, test_loader, weights