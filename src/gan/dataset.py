import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CustomImageDataset(Dataset):
    """Dataset for loading images from the data directory."""
    
    def __init__(self, data_dir, image_size=256, train=True):
        self.data_dir = data_dir
        self.image_size = image_size
        self.train = train
        
        self.image_paths = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root, file))
        
        split_idx = int(0.8 * len(self.image_paths))
        if train:
            self.image_paths = self.image_paths#[:split_idx]
        else:
            self.image_paths = self.image_paths[split_idx:]
    
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return torch.randn(3, self.image_size, self.image_size)
