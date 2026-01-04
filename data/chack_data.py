import json
import os
from collections import Counter


def analyze_pisc_dist(data_root, split='train'):
    # Dosya yolları
    split_file = os.path.join(data_root, 'relationship_split', f'relation_{split}idx.json')
    rel_file = os.path.join(data_root, 'relationship.json')

    if not os.path.exists(split_file) or not os.path.exists(rel_file):
        print(f"❌ Hata: {data_root} içinde JSON dosyaları bulunamadı!")
        return

    with open(split_file, 'r') as f:
        active_ids = [str(i) for i in json.load(f)]
    with open(rel_file, 'r') as f:
        rel_data = json.load(f)

    # PISC Standart İsim Listesi (ID 1-6 sırasına göre)
    class_names = [
        "Friends",  # ID 1 -> İndeks 0
        "Family",  # ID 2 -> İndeks 1
        "Couple",  # ID 3 -> İndeks 2
        "Professional",  # ID 4 -> İndeks 3
        "Commercial",  # ID 5 -> İndeks 4
        "No Relation"  # ID 6 -> İndeks 5
    ]

    label_counts = Counter()
    total_pairs = 0

    for img_id in active_ids:
        if img_id in rel_data:
            img_pairs = rel_data[img_id]
            for pair_key, label_idx in img_pairs.items():
                val = int(label_idx)

                # --- DÜZELTME NOKTASI ---
                # PISC etiketleri 1'den başlar (1,2,3,4,5,6).
                # Python listeleri 0'dan başlar. Bu yüzden 1 çıkartıyoruz.
                idx = val - 1

                # Güvenlik kontrolü: Sadece 0-5 arası indeksleri say
                if 0 <= idx <= 5:
                    label_counts[idx] += 1
                    total_pairs += 1

    print(f"\n📊 --- {split.upper()} SETİ GERÇEK ANALİZİ ---")
    print(f"🖼️  Toplam Resim: {len(active_ids)}")
    print(f"👥 Toplam Çift : {total_pairs}")
    print("-" * 45)
    print(f"{'SINIF ADI':<15} | {'ADET':<8} | {'YÜZDE'}")
    print("-" * 45)

    for i, name in enumerate(class_names):
        count = label_counts[i]
        percentage = (count / total_pairs) * 100 if total_pairs > 0 else 0
        print(f"{name:15} | {count:8} | %{percentage:.2f}")
    print("-" * 45)


if __name__ == "__main__":
    # Klasör yolunun doğruluğundan emin ol
    DATA_PATH = 'dataset'
    analyze_pisc_dist(DATA_PATH, 'train')
    analyze_pisc_dist(DATA_PATH, 'val')