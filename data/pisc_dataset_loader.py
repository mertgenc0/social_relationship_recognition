import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import random


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
        # Fiziksel klasör ismin 'images' ise burayı 'images' yapmalısın
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

        self.pairs = []
        for img_id in active_ids:
            img_id_str = str(img_id)
            if img_id_str in rel_data and img_id_str in info_dict:
                img_pairs = rel_data[img_id_str]
                for pair_key, label_idx in img_pairs.items():
                    self.pairs.append({
                        'filename': f"{img_id_str}.jpg",
                        'label': int(label_idx),
                        'caption': f"Two people in a {self.label_names[min(int(label_idx), 5)]} relationship."
                    })

        self.transform = transform
        print(f"✅ {split.upper()} Yüklendi: {len(self.pairs)} çift bulundu.")

        # Rastgele ama sınıfları koruyan bir seçim (örnek)
        if split == 'train':
            random.shuffle(self.pairs)
            self.pairs = self.pairs[:10000]  # İlk 10.000 örneği al

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair_data = self.pairs[idx]
        img_path = os.path.join(self.image_dir, pair_data['filename'])

        # 1. Image Referansını Sağlamlaştır
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')

        # 2. Transformları Uygula
        if self.transform:
            image = self.transform(image)

        # 3. ETİKET DÜZELTME VE GÜVENLİK (Kritik Nokta!)
        # PISC etiketleri 1,2,3,4,5,6 şeklindedir.
        # PyTorch indeksleme için 0,1,2,3,4,5 bekler.
        # Bu yüzden '1 çıkartıyoruz'.
        raw_label = int(pair_data['label'])
        clean_label = max(0, min(raw_label - 1, 5))

        # 4. Trainer ile %100 Uyumlu Dönüş
        # NOT: trainer.py 'image' (tekil) beklediği için anahtar 'image' olmalı.
        return {
            'image': image,
            'caption': pair_data['caption'],
            'label': torch.tensor(clean_label, dtype=torch.long)
        }


def get_pisc_dataloaders(data_root, batch_size=4, num_workers=0):
    # EĞİTİM İÇİN VERİ ARTIRIMI (Overfitting'e karşı)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = PISCDataset(data_root, split='train', transform=train_transform)
    val_ds = PISCDataset(data_root, split='val', transform=val_transform)

    # Ağırlıkları sadece eğitim setinden hesapla
    weights = calculate_class_weights(train_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, weights