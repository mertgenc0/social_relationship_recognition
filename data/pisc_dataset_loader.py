import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class PISCDataset(Dataset):
    def __init__(self, data_root, split='train', transform=None):
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, 'image')

        # Dosya yolları
        split_file = os.path.join(data_root, 'relationship_split', f'relation_{split}idx.json')
        rel_file = os.path.join(data_root, 'relationship.json')
        info_file = os.path.join(data_root, 'annotation_image_info.json')

        # 1. Verileri Yükle
        with open(split_file, 'r') as f:
            active_ids = json.load(f)  # Liste ['8408', ...]
        with open(rel_file, 'r') as f:
            rel_data = json.load(f)  # Sözlük {'13357': {'1 2': 4}}
        with open(info_file, 'r') as f:
            info_list = json.load(f)  # Liste [{'id': 0, ...}]

        # 2. Hızlı erişim için info listesini sözlüğe çevir (ID -> Data)
        info_dict = {str(item['id']): item for item in info_list}

        # Etiket isimleri (Rapordaki sıralamaya göre)
        self.label_names = ["Friends", "Family", "Couple", "Professional", "Commercial", "No Relation"]

        self.pairs = []
        # 3. Verileri Eşleştir
        for img_id in active_ids:
            img_id_str = str(img_id)
            if img_id_str in rel_data and img_id_str in info_dict:
                # İlişkileri oku (Örn: {'1 2': 4})
                img_pairs = rel_data[img_id_str]
                for pair_key, label_idx in img_pairs.items():
                    # Dosya adını oluştur (PISC genellikle id.jpg kullanır)
                    filename = f"{img_id_str}.jpg"

                    self.pairs.append({
                        'filename': filename,
                        'label': int(label_idx),
                        'caption': f"Two people in a {self.label_names[min(int(label_idx), 5)]} relationship."
                    })

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(f"✅ {split.upper()} Yüklendi: {len(self.pairs)} çift bulundu.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair_data = self.pairs[idx]
        img_path = os.path.join(self.image_dir, pair_data['filename'])

        # Görüntü yükleme
        if not os.path.exists(img_path):
            image = Image.new('RGB', (224, 224), color='gray')
        else:
            image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # trainer.py'deki Satır 103, 104 ve 105 ile BİREBİR UYUM:
        return {
            'image': image,  # Trainer 'image' (çoğul) bekliyor [Satır 103]
            'caption': pair_data['caption'],  # Trainer 'caption' (tekil) bekliyor [Satır 104]
            'label': torch.tensor(pair_data['label'], dtype=torch.long)  # Trainer 'label' (tekil) bekliyor [Satır 105]
        }


def get_pisc_dataloaders(data_root, batch_size=4, num_workers=0):
    """
    M2 MacBook Air için optimize edilmiş yükleyici.
    num_workers parametresi eklendi.
    """
    train_ds = PISCDataset(data_root, split='train')
    val_ds = PISCDataset(data_root, split='val')

    # DataLoader içinde num_workers'ı kullanıyoruz
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Klasör yapına göre: data/pisc_dataset_loader.py ve data/dataset/
    DATA_ROOT = os.path.join(os.path.dirname(__file__), 'dataset')

    try:
        train_loader, _ = get_pisc_dataloaders(DATA_ROOT)
        batch = next(iter(train_loader))
        print(f"\n🚀 BAŞARILI! İlk Batch Yüklendi. Image Shape: {batch['image'].shape}")
    except Exception as e:
        print(f"\n❌ Hata Devam Ediyor: {e}")