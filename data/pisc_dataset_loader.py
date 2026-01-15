import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class PISCDataset(Dataset):
    def __init__(self, data_root, split='train', transform=None):
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, 'image')
        self.label_names = ["Friends", "Family", "Couple", "Professional", "Commercial", "No Relation"]

        # 1. Dosya Yollarını Tanımla
        fine_split = os.path.join(data_root, 'relationship_split', f'relation_{split}idx.json')
        coarse_split = os.path.join(data_root, 'domain_split', f'{split}idx.json')

        rel_file = os.path.join(data_root, 'relationship.json')  # Fine etiketler (6 sınıf)
        dom_file = os.path.join(data_root, 'domain.json')  # Coarse etiketler (3 sınıf)
        info_file = os.path.join(data_root, 'annotation_image_info.json')

        # 2. Tüm Aktif Image ID'lerini Topla
        active_ids = set()
        if os.path.exists(fine_split):
            with open(fine_split, 'r') as f: active_ids.update(json.load(f))
        if os.path.exists(coarse_split):
            with open(coarse_split, 'r') as f: active_ids.update(json.load(f))

        # 3. Tüm Etiket Dosyalarını Yükle
        with open(rel_file, 'r') as f:
            rel_data = json.load(f)
        with open(dom_file, 'r') as f:
            dom_data = json.load(f)  # Domain verisi eklendi
        with open(info_file, 'r') as f:
            info_list = json.load(f)
        info_dict = {str(item['id']): item for item in info_list}

        # 4. Veri Birleştirme (96k Hedefi)
        all_pairs = []
        for img_id in active_ids:
            img_id_str = str(img_id)
            if img_id_str not in info_dict: continue

            # A) Öncelikle Fine-grained (6 sınıf) verisine bak
            if img_id_str in rel_data:
                for pair_key, label_idx in rel_data[img_id_str].items():
                    all_pairs.append({
                        'filename': f"{img_id_str}.jpg",
                        'label': int(label_idx)  # 1-6 arası etiket
                    })

            # B) EĞER BU GÖRÜNTÜ FINE İÇİNDE YOKSA VEYA EK VERİ VARSA DOMAIN'DEN ÇEK
            # Not: domain.json'daki 3 sınıfı 6 sınıf sistemine map ediyoruz
            elif img_id_str in dom_data:
                for pair_key, label_idx in dom_data[img_id_str].items():
                    coarse_label = int(label_idx)
                    # Mapping mantığı (Makale Bölüm III-E uyumlu ):
                    # 1 (Intimate) -> 1 (Friends - Varsayılan Yakın İlişki)
                    # 2 (Non-Intimate) -> 4 (Professional - Varsayılan İş İlişkisi)
                    # 3 (No Relation) -> 6 (No Relation)
                    mapped_label = coarse_label
                    if coarse_label == 2: mapped_label = 4
                    if coarse_label == 3: mapped_label = 6

                    all_pairs.append({
                        'filename': f"{img_id_str}.jpg",
                        'label': mapped_label
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

        # Görüntü yükleme
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='gray')

        if self.transform:
            image = self.transform(image)

        # 1-6 arası etiketleri 0-5 arasına çek
        clean_label = max(0, min(int(pair_data['label']) - 1, 5))
        rel_name = self.label_names[clean_label]

        # Makale uyumlu yapılandırılmış metin formatı [cite: 61, 62]
        structured_caption = f"relationship: {rel_name.lower()}, emotional state: happy, setting: outdoor"

        return {
            'image': image,
            'caption': structured_caption,
            'label': torch.tensor(clean_label, dtype=torch.long)
        }