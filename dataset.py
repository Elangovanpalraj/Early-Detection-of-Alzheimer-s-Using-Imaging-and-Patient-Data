import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class AlzheimerDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = os.path.join(root_dir, mode)
        self.transform = transform
        self.image_files = []
        self.labels = []

        # Define class labels based on folder names
        self.class_map = {
            'Mild Impairment': 0,
            'Moderate Impairment': 1,
            'No Impairment': 2,
            'Very Mild Impairment': 3
        }

        # Iterate over class folders
        for category in self.class_map.keys():
            category_path = os.path.join(self.root_dir, category)
            for file in os.listdir(category_path):
                if file.endswith('.jpg') or file.endswith('.png'):
                    self.image_files.append(os.path.join(category_path, file))
                    self.labels.append(self.class_map[category])

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure 3 channels
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
