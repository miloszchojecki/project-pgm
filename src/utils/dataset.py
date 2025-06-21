from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

class FidClassDataset(Dataset):
    def __init__(self, image_dir, image_size):
        self.image_paths = list(Path(image_dir).glob("*"))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB") 
        img = self.transform(img)
        return img

class SingleClassDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = list(Path(folder).glob("*.jpg"))
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img
    
    
class MelDataset(Dataset):
    def __init__(self, data_path, transform, mel_class_idx: int):
        self.paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            self.paths.extend(Path(data_path).glob(ext))
        self.paths = [str(p) for p in self.paths]
        self.transform = transform
        self.label = mel_class_idx
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.paths[idx]).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, self.label
        except Exception as e:
            print(f"Error loading {self.paths[idx]}: {e}")
            return torch.randn(3, 256, 256), self.label