import json
import os
from collections import Counter


def kesif_analizi_yap(data_root):
    rel_file = os.path.join(data_root, 'relationship.json')

    if not os.path.exists(rel_file):
        print(f"❌ Hata: {rel_file} bulunamadı!")
        return

    # Sadece ilişki verisini yükle (tüm veri setini taramak için)
    with open(rel_file, 'r') as f:
        rel_data = json.load(f)

    # Tüm etiketleri topla
    tum_etiketler = []
    for img_id in rel_data:
        img_pairs = rel_data[img_id]
        for pair_key, label_idx in img_pairs.items():
            tum_etiketler.append(int(label_idx))

    # Benzersiz etiketleri ve sayılarını hesapla
    sayici = Counter(tum_etiketler)
    sirali_etiketler = sorted(sayici.keys())
    toplam_ornek = len(tum_etiketler)

    print("\n🔍 --- VERİ SETİ ETİKET KEŞFİ ---")
    print(f"📦 Toplam İlişki Örneği: {toplam_ornek}")
    print(f"🔢 Bulunan Benzersiz Etiket Sayısı: {len(sirali_etiketler)}")
    print("-" * 45)
    print(f"{'ETİKET ID':<10} | {'ADET':<10} | {'YÜZDE'}")
    print("-" * 45)

    for etiket in sirali_etiketler:
        adet = sayici[etiket]
        yuzde = (adet / toplam_ornek) * 100
        print(f"ID: {etiket:<6} | {adet:<10} | %{yuzde:.2f}")

    print("-" * 45)
    print("💡 Not: Eğer 0-5 dışındaki ID'ler çok azsa, bunlar hatalı veri olabilir.")


if __name__ == "__main__":
    DATA_PATH = 'dataset'  # Kendi yolunu kontrol et
    kesif_analizi_yap(DATA_PATH)