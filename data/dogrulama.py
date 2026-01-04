import json
import os


def etiketleri_dogrula(data_root):
    rel_file = os.path.join(data_root, 'relationship.json')
    with open(rel_file, 'r') as f:
        rel_data = json.load(f)

    # Her ID'den örnek resim bulalım
    ornekler = {str(i): None for i in range(1, 7)}

    for img_id, pairs in rel_data.items():
        for pair_key, label_idx in pairs.items():
            lbl = str(label_idx)
            if lbl in ornekler and ornekler[lbl] is None:
                ornekler[lbl] = img_id

    print("\n🧐 --- GÖRSEL DOĞRULAMA LİSTESİ ---")
    print("Aşağıdaki resimleri 'data/dataset/image' klasöründe aç ve bak:")
    print("-" * 50)
    mapping = {
        "1": "Friends (Arkadaş mı?)",
        "2": "Family (Aile mi?)",
        "3": "Couple (Sevgili mi?)",
        "4": "Professional (İş arkadaşı mı?)",
        "5": "Commercial (Müşteri/Satıcı mı?)",
        "6": "No Relation (Tanımıyor mu?)"
    }

    for lbl, img_name in ornekler.items():
        if img_name:
            print(f"ID {lbl} [{mapping[lbl]}]: {img_name}.jpg")
    print("-" * 50)


if __name__ == "__main__":
    etiketleri_dogrula('dataset')