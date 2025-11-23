import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class PISCDataset(Dataset):
    """
    PISC Dataset Loader for Social Relationship Recognition

    PISC contains images with annotated relationship pairs:
    - Coarse-grained: Intimate, Non-intimate, No-relation
    - Fine-grained: Friends, Family, Couple, Professional, Commercial, No-relation
    """

    def __init__(self, data_dir, split='train', task='fine', transform=None):
        """
        Args:
            data_dir: Path to data/processed/
            split: 'train', 'val', or 'test'
            task: 'coarse' or 'fine' (relationship granularity)
            transform: Image transformations
        """
        self.data_dir = os.path.join(data_dir, split)
        self.split = split
        self.task = task

        # Relationship labels - ÖNCE TANIMLA
        self.coarse_labels = {
            'intimate': 0,
            'non-intimate': 1,
            'no-relation': 2
        }

        self.fine_labels = {
            'friends': 0,
            'family': 1,
            'couple': 2,
            'professional': 3,
            'commercial': 4,
            'no-relation': 5
        }

        # Default image transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet means
                    std=[0.229, 0.224, 0.225]  # ImageNet stds
                )
            ])
        else:
            self.transform = transform

        # Load annotations (will be created after dataset download)
        annotations_path = os.path.join(self.data_dir, 'annotations.json')

        # Initialize empty data structures
        self.annotations = {}
        self.pairs = []

        if not os.path.exists(annotations_path):
            print(f"⚠️  Annotations not found at {annotations_path}")
            print("   Please download PISC dataset first!")
            return

        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)

        # Create list of all relationship pairs
        for img_name, img_data in self.annotations.items():
            if 'pairs' not in img_data:
                continue

            for pair_id, pair_data in img_data['pairs'].items():
                self.pairs.append({
                    'image_name': img_name,
                    'pair_id': pair_id,
                    'coarse_label': pair_data.get('coarse_label', 'no-relation'),
                    'fine_label': pair_data.get('fine_label', 'no-relation'),
                    'caption': pair_data.get('caption', ''),
                    'person1_bbox': pair_data.get('person1_bbox', [0, 0, 224, 224]),
                    'person2_bbox': pair_data.get('person2_bbox', [0, 0, 224, 224])
                })

        print(f"📊 {split.upper()} dataset: {len(self.pairs)} relationship pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - image: Transformed image tensor [3, 224, 224]
                - caption: Text description (string)
                - coarse_label: Coarse relationship label (int)
                - fine_label: Fine relationship label (int)
                - image_name: Image filename
                - pair_id: Unique pair identifier
        """
        pair_data = self.pairs[idx]

        # Load image
        img_path = os.path.join(self.data_dir, pair_data['image_name'])

        if not os.path.exists(img_path):
            # Return dummy data if image not found (for testing)
            print(f"⚠️  Image not found: {img_path}")
            image = Image.new('RGB', (224, 224), color='gray')
        else:
            image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Convert labels to integers
        coarse_label = self.coarse_labels.get(
            pair_data['coarse_label'],
            self.coarse_labels['no-relation']
        )
        fine_label = self.fine_labels.get(
            pair_data['fine_label'],
            self.fine_labels['no-relation']
        )

        # Select label based on task
        if self.task == 'coarse':
            label = coarse_label
        else:  # fine
            label = fine_label

        return {
            'image': image,
            'caption': pair_data['caption'],
            'label': torch.tensor(label, dtype=torch.long),
            'coarse_label': torch.tensor(coarse_label, dtype=torch.long),
            'fine_label': torch.tensor(fine_label, dtype=torch.long),
            'image_name': pair_data['image_name'],
            'pair_id': pair_data['pair_id']
        }

    def get_num_classes(self):
        """Return number of classes for current task"""
        if self.task == 'coarse':
            return len(self.coarse_labels)
        else:
            return len(self.fine_labels)


# Test code
if __name__ == "__main__":
    print("🧪 Testing PISC Dataset Loader...")
    print("=" * 60)

    # Create dataset (will show warning if data not ready)
    dataset = PISCDataset(
        data_dir='data/processed',
        split='train',
        task='fine'
    )

    print(f"\n📈 Dataset Statistics:")
    print(f"   Total pairs: {len(dataset)}")
    print(f"   Num classes: {dataset.get_num_classes()}")

    if len(dataset) > 0:
        print(f"\n🔍 Sample Data:")
        sample = dataset[0]
        print(f"   Image shape: {sample['image'].shape}")
        print(f"   Caption: {sample['caption'][:50]}...")
        print(f"   Label: {sample['label'].item()}")
        print(f"   Fine label: {sample['fine_label'].item()}")
        print(f"   Coarse label: {sample['coarse_label'].item()}")

        print(f"\n✅ Dataset loader working correctly!")
    else:
        print(f"\n⚠️  No data found. Please:")
        print(f"   1. Download PISC dataset")
        print(f"   2. Place in data/raw/")
        print(f"   3. Run preprocessing script")

    print("=" * 60)