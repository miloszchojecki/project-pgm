from typing import List, Tuple
import os
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
from config import Config

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, image_size=256):
        self.image_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(data_dir)
            for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.randn(3, 256, 256)

class MelDataset(Dataset):
    def __init__(self, gan_dir: str, transform: transforms.Compose, mel_class_idx: int):
        self.paths = [
            str(p) for p in Path(gan_dir).glob("*.[jp][pn][gf]")  # Matches jpg, jpeg, png
        ]
        self.transform = transform
        self.label = mel_class_idx
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        try:
            image = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(image), self.label
        except Exception as e:
            print(f"Error loading {self.paths[idx]}: {e}")
            return torch.randn(3, 256, 256), self.label

def get_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

def get_dataloaders(config: Config, model_type: str) -> Tuple[DataLoader, DataLoader, dict]:
    transform = get_transforms(config.image_size)
    
    train_dataset_real = datasets.ImageFolder(
        root=str(config.dataset_paths["real_train"]), 
        transform=transform
    )
    val_dataset = datasets.ImageFolder(
        root=str(config.dataset_paths["val"]), 
        transform=transform
    )
    
    class_to_idx = train_dataset_real.class_to_idx
    
    if model_type in ["diffusion", "gan", "vae"]:
        mel_class_idx = class_to_idx.get("mel", 0)
        gan_dataset = MelDataset(
            gan_dir=str(config.get_gen_path(model_type)),
            transform=transform,
            mel_class_idx=mel_class_idx
        )
        train_dataset = ConcatDataset([train_dataset_real, gan_dataset])
        print(f"GAN data added: {len(gan_dataset)} samples")
    elif model_type == "base":
        train_dataset = train_dataset_real
        print("Using only real data for training")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    return train_loader, val_loader, class_to_idx
