"""
PISC Dataset Preprocessing Script
Converts raw PISC dataset to the format expected by PISCDataset loader

Steps:
1. Download dataset from Zenodo: https://zenodo.org/records/1059155
2. Extract image and place JSON files in data/raw/
3. Run this script to create processed annotations

Usage:
    python scripts/preprocess_pisc.py
"""

import os
import json
from collections import defaultdict
from tqdm import tqdm
import shutil


class PISCPreprocessor:
    def __init__(self, raw_dir='data/raw', processed_dir='data/processed'):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir

        # PISC dosya yolları
        self.img_info_path = os.path.join(raw_dir, 'annotation_image_info.json')
        self.domain_path = os.path.join(raw_dir, 'domain.json')
        self.relation_path = os.path.join(raw_dir, 'relationship.json')
        self.relation_split_path = os.path.join(raw_dir, 'relationship_split')

        # Label mapping
        self.domain_to_coarse = {
            1: 'intimate',
            2: 'non-intimate',
            3: 'no-relation'
        }

        self.relation_to_fine = {
            1: 'friends',
            2: 'family',
            3: 'couple',
            4: 'professional',
            5: 'commercial',
            6: 'no-relation'
        }

    def load_json(self, filepath):
        """JSON dosyasını yükle"""
        print(f"📖 Loading {os.path.basename(filepath)}...")
        with open(filepath, 'r') as f:
            return json.load(f)

    def load_split_file(self, filepath):
        """Split dosyasını yükle (json array formatında)"""
        # relation_trainidx.json, relation_validx.json, relation_testidx.json
        json_path = filepath.replace('.txt', 'idx.json')

        if os.path.exists(json_path):
            # JSON format (PISC dataset formatı)
            with open(json_path, 'r') as f:
                data = json.load(f)
                # JSON array içindeki string ID'leri döndür
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return list(data.keys())
        elif os.path.exists(filepath):
            # Text format (alternatif)
            with open(filepath, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        else:
            raise FileNotFoundError(f"Split file not found: {json_path} or {filepath}")

    def get_image_filename(self, image_id, img_info):
        """Image ID'den dosya adını al"""
        if image_id in img_info:
            # Flickr ID'den filename oluştur
            flickr_id = img_info[image_id].get('flickr_id', image_id)
            return f"{flickr_id}.jpg"
        return f"{image_id}.jpg"

    def process_dataset(self):
        """Ana preprocessing fonksiyonu"""
        print("=" * 80)
        print("🚀 PISC Dataset Preprocessing Started")
        print("=" * 80)

        # 1. Dosyaları kontrol et
        if not self.check_raw_files():
            return False

        # 2. JSON dosyalarını yükle
        img_info = self.load_json(self.img_info_path)
        domain_data = self.load_json(self.domain_path)
        relation_data = self.load_json(self.relation_path)

        # 3. Split dosyalarını yükle
        print("\n📂 Loading dataset splits...")

        # PISC dataset split dosyaları relation_trainidx.json formatında
        train_path = os.path.join(self.relation_split_path, 'relation_train.txt')
        val_path = os.path.join(self.relation_split_path, 'relation_val.txt')
        test_path = os.path.join(self.relation_split_path, 'relation_test.txt')

        train_ids = self.load_split_file(train_path)
        val_ids = self.load_split_file(val_path)
        test_ids = self.load_split_file(test_path)

        print(f"   Train: {len(train_ids)} image")
        print(f"   Val: {len(val_ids)} image")
        print(f"   Test: {len(test_ids)} image")

        # 4. Her split için annotations oluştur
        splits = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }

        for split_name, image_ids in splits.items():
            print(f"\n🔄 Processing {split_name.upper()} split...")
            self.process_split(
                split_name,
                image_ids,
                img_info,
                domain_data,
                relation_data
            )

        print("\n" + "=" * 80)
        print("✅ Preprocessing completed successfully!")
        print("=" * 80)
        print("\n📌 Next steps:")
        print("   1. Run: python scripts/organize_images.py")
        print("   2. Test with: python data/pisc_dataset_loader_old.py")

        return True

    def process_split(self, split_name, image_ids, img_info, domain_data, relation_data):
        """Belirli bir split'i işle"""
        split_dir = os.path.join(self.processed_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        annotations = {}
        pair_count = 0

        # Debug: İlk birkaç ID'yi yazdır
        print(f"\n   🔍 Debug - First 5 image IDs from split file:")
        for i, img_id in enumerate(image_ids[:5]):
            print(f"      {i + 1}. '{img_id}' (type: {type(img_id).__name__})")

        print(f"\n   🔍 Debug - img_info type: {type(img_info).__name__}")
        print(f"   🔍 Debug - img_info length: {len(img_info)}")

        # img_info dict ise
        if isinstance(img_info, dict):
            print(f"\n   🔍 Debug - First 5 image IDs from img_info (dict keys):")
            for i, img_id in enumerate(list(img_info.keys())[:5]):
                print(f"      {i + 1}. '{img_id}' (type: {type(img_id).__name__})")
        # img_info list ise
        elif isinstance(img_info, list):
            print(f"\n   🔍 Debug - First 5 items from img_info (list):")
            for i, item in enumerate(img_info[:5]):
                if isinstance(item, dict) and 'image_id' in item:
                    print(
                        f"      {i + 1}. image_id='{item.get('image_id')}' (type: {type(item.get('image_id')).__name__})")
                else:
                    print(
                        f"      {i + 1}. {type(item).__name__} - keys: {list(item.keys())[:5] if isinstance(item, dict) else 'N/A'}")

        # img_info'yu dict'e dönüştür (eğer list ise)
        if isinstance(img_info, list):
            print(f"\n   ⚙️  Converting img_info from list to dict...")
            img_info_dict = {}
            for item in img_info:
                if isinstance(item, dict):
                    img_id = item.get('image_id')
                    if img_id is not None:
                        # ID'yi string'e çevir
                        img_info_dict[str(img_id)] = item
            img_info = img_info_dict
            print(f"   ✓ Converted {len(img_info)} items")

        found_count = 0
        not_found_count = 0

        for img_id in tqdm(image_ids, desc=f"Processing {split_name}"):
            # ID'yi string'e çevir (emin olmak için)
            img_id = str(img_id)

            # Image bilgilerini al
            if img_id not in img_info:
                not_found_count += 1
                if not_found_count <= 3:  # İlk 3 eksik ID'yi göster
                    print(f"\n   ⚠️  Image ID not found in img_info: '{img_id}'")
                continue

            found_count += 1
            img_data = img_info[img_id]
            filename = self.get_image_filename(img_id, img_info)

            # Person bounding boxes
            persons = img_data.get('annotations', [])

            if len(persons) < 2:
                continue  # En az 2 kişi olmalı

            # Relationship pairs oluştur
            pairs = {}

            # Domain (coarse) ve relation (fine) annotations'ı al
            img_domain = domain_data.get(img_id, {})
            img_relation = relation_data.get(img_id, {})

            # Tüm kişi çiftlerini işle
            for i in range(len(persons)):
                for j in range(i + 1, len(persons)):
                    person1 = persons[i]
                    person2 = persons[j]

                    pair_key = f"{i}_{j}"

                    # Coarse label (domain)
                    domain_label = img_domain.get(pair_key, 3)  # Default: no-relation
                    coarse_label = self.domain_to_coarse.get(domain_label, 'no-relation')

                    # Fine label (relation)
                    relation_label = img_relation.get(pair_key, 6)  # Default: no-relation
                    fine_label = self.relation_to_fine.get(relation_label, 'no-relation')

                    pairs[pair_key] = {
                        'coarse_label': coarse_label,
                        'fine_label': fine_label,
                        'person1_bbox': person1.get('bbox', [0, 0, 100, 100]),
                        'person2_bbox': person2.get('bbox', [0, 0, 100, 100]),
                        'caption': ''  # PISC'de caption yok, boş bırak
                    }

                    pair_count += 1

            if pairs:
                annotations[filename] = {
                    'image_id': img_id,
                    'width': img_data.get('width', 0),
                    'height': img_data.get('height', 0),
                    'pairs': pairs
                }

        print(f"\n   📊 Stats: Found {found_count} image, Not found {not_found_count} image")

        # Annotations'ı kaydet
        output_path = os.path.join(split_dir, 'annotations.json')
        with open(output_path, 'w') as f:
            json.dump(annotations, f, indent=2)

        print(f"   ✓ Created {len(annotations)} image with {pair_count} pairs")
        print(f"   ✓ Saved to: {output_path}")

    def check_raw_files(self):
        """Gerekli dosyaların varlığını kontrol et"""
        print("🔍 Checking raw data files...\n")

        required_files = [
            self.img_info_path,
            self.domain_path,
            self.relation_path,
        ]

        required_dirs = [
            self.relation_split_path,
        ]

        missing = []

        for filepath in required_files:
            if os.path.exists(filepath):
                print(f"   ✓ {os.path.basename(filepath)}")
            else:
                print(f"   ✗ {os.path.basename(filepath)} - MISSING")
                missing.append(filepath)

        for dirpath in required_dirs:
            if os.path.exists(dirpath):
                print(f"   ✓ {os.path.basename(dirpath)}/")
            else:
                print(f"   ✗ {os.path.basename(dirpath)}/ - MISSING")
                missing.append(dirpath)

        if missing:
            print("\n❌ Missing required files!")
            print("\n📥 Please download PISC dataset from:")
            print("   https://zenodo.org/records/1059155")
            print("\n📋 Required files:")
            print("   - annotation_image_info.json")
            print("   - domain.json")
            print("   - relationship.json")
            print("   - relationship_split/ folder (with train.txt, val.txt, test.txt)")
            print("   - image/ folder (extracted from image-* archives)")
            print(f"\n📂 Current directory: {os.path.abspath(self.raw_dir)}")
            print(f"📂 Files in raw dir:")
            try:
                for item in os.listdir(self.raw_dir):
                    item_path = os.path.join(self.raw_dir, item)
                    if os.path.isdir(item_path):
                        print(f"   📁 {item}/")
                    else:
                        print(f"   📄 {item}")
            except Exception as e:
                print(f"   Error listing directory: {e}")
            return False

        print("\n✅ All required files found!")
        return True


def main():
    """Ana fonksiyon"""
    # Script'in bulunduğu dizinden proje root'unu bul
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # scripts/ klasöründen bir üst dizin

    raw_dir = os.path.join(project_root, 'data', 'raw')
    processed_dir = os.path.join(project_root, 'data', 'processed')

    print(f"📂 Project root: {project_root}")
    print(f"📂 Raw data dir: {raw_dir}")
    print(f"📂 Processed dir: {processed_dir}\n")

    preprocessor = PISCPreprocessor(
        raw_dir=raw_dir,
        processed_dir=processed_dir
    )

    success = preprocessor.process_dataset()

    if not success:
        print("\n" + "=" * 80)
        print("💡 TIP: Check if files are in the correct location")
        print("=" * 80)


if __name__ == "__main__":
    main()