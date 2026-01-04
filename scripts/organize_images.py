"""
PISC Image Organizer
Görüntüleri annotations.json'a göre train/val/test klasörlerine kopyalar

Usage:
    python scripts/organize_images.py
"""

import os
import json
import shutil
from tqdm import tqdm


def organize_images(raw_images_dir='data/raw/image',
                    processed_dir='data/processed'):
    """
    Görüntüleri annotations'a göre organize et
    """
    print("=" * 80)
    print("🗂️  PISC Image Organization")
    print("=" * 80)

    # Kaynak görüntü klasörünü kontrol et
    if not os.path.exists(raw_images_dir):
        print(f"\n❌ Error: Image directory not found: {raw_images_dir}")
        print("\n📥 Please:")
        print("   1. Download image from Zenodo")
        print("   2. Extract with: cat image-* | tar zx")
        print("   3. Place in data/raw/image/")
        return False

    splits = ['train', 'val', 'test']
    total_copied = 0
    total_missing = 0

    for split in splits:
        print(f"\n📂 Processing {split.upper()} split...")

        # Annotations dosyasını yükle
        annotations_path = os.path.join(processed_dir, split, 'annotations.json')

        if not os.path.exists(annotations_path):
            print(f"   ⚠️  Annotations not found: {annotations_path}")
            print(f"   ℹ️  Run preprocess_pisc.py first!")
            continue

        with open(annotations_path, 'r') as f:
            annotations = json.load(f)

        # Hedef klasörü oluştur
        target_dir = os.path.join(processed_dir, split)
        os.makedirs(target_dir, exist_ok=True)

        copied = 0
        missing = 0

        # Her görüntüyü kopyala
        for image_name in tqdm(annotations.keys(), desc=f"Copying {split} image"):
            source_path = os.path.join(raw_images_dir, image_name)
            target_path = os.path.join(target_dir, image_name)

            # Kaynak dosya var mı?
            if not os.path.exists(source_path):
                missing += 1
                # Alternatif uzantıları dene
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG']:
                    alt_source = source_path.replace('.jpg', ext)
                    if os.path.exists(alt_source):
                        source_path = alt_source
                        break
                else:
                    # print(f"   ⚠️  Missing: {image_name}")
                    continue

            # Hedef dosya zaten var mı?
            if os.path.exists(target_path):
                copied += 1
                continue

            # Dosyayı kopyala
            try:
                shutil.copy2(source_path, target_path)
                copied += 1
            except Exception as e:
                print(f"   ❌ Error copying {image_name}: {e}")
                missing += 1

        print(f"   ✓ Copied: {copied} image")
        if missing > 0:
            print(f"   ⚠️  Missing: {missing} image")

        total_copied += copied
        total_missing += missing

    # Özet
    print("\n" + "=" * 80)
    print("📊 Summary")
    print("=" * 80)
    print(f"Total image copied: {total_copied}")
    print(f"Total image missing: {total_missing}")

    if total_missing > 0:
        print("\n⚠️  Some image are missing!")
        print("   This might be normal if the dataset has been updated.")

    if total_copied > 0:
        print("\n✅ Image organization completed!")
        print("\n📌 Next step:")
        print("   Test the dataset loader:")
        print("   python data/pisc_dataset_loader.py")

    return True


def check_image_formats(raw_images_dir='data/raw/image'):
    """
    Görüntü formatlarını kontrol et
    """
    print("\n🔍 Checking image formats...")

    if not os.path.exists(raw_images_dir):
        print(f"   ❌ Directory not found: {raw_images_dir}")
        return

    extensions = {}
    sample_files = []

    for filename in os.listdir(raw_images_dir)[:100]:  # İlk 100 dosya
        _, ext = os.path.splitext(filename)
        extensions[ext] = extensions.get(ext, 0) + 1
        if len(sample_files) < 5:
            sample_files.append(filename)

    print("\n   Found file extensions:")
    for ext, count in sorted(extensions.items()):
        print(f"      {ext}: {count} files")

    print("\n   Sample filenames:")
    for filename in sample_files:
        print(f"      {filename}")


def main():
    """Ana fonksiyon"""
    import argparse

    parser = argparse.ArgumentParser(description='Organize PISC image')
    parser.add_argument('--raw-dir', default='data/raw/image',
                        help='Raw image directory')
    parser.add_argument('--processed-dir', default='data/processed',
                        help='Processed data directory')
    parser.add_argument('--check', action='store_true',
                        help='Only check image formats without copying')

    args = parser.parse_args()

    if args.check:
        check_image_formats(args.raw_dir)
    else:
        organize_images(args.raw_dir, args.processed_dir)


if __name__ == "__main__":
    main()