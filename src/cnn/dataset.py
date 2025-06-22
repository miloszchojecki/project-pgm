from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets
from utils.transform import create_transformation
from utils.dataset import MelDataset


class RemappedImageFolder(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_to_idx = {'other': 0, 'mel': 1}
        self.targets = [self.class_to_idx[self.classes[t]] for t in self.targets]


def get_dataloaders(cfg, model_type):
    image_size = int(cfg.train.cnn.image_size)
    
    transform = create_transformation(image_size)
    train_path = Path(cfg.paths.train)
    test_path = Path(cfg.paths.test)
    
    train_dataset_real = RemappedImageFolder(
        root=str(train_path), 
        transform=transform
    )
    print(f"Path to training data: {train_path}")
    val_dataset = RemappedImageFolder(
        root=str(test_path), 
        transform=transform
    )
    
    class_to_idx = train_dataset_real.class_to_idx
    
    if model_type in ["diffusion", "gan", "vae"]:
        mel_class_idx = class_to_idx.get("mel", 0)
        
        gan_dataset = MelDataset(
            data_path=str(cfg.generate[model_type].result_path),
            transform=transform,
            mel_class_idx=mel_class_idx
        )
        train_dataset = ConcatDataset([train_dataset_real, gan_dataset])
        print(f"{model_type} data added: {len(gan_dataset)} samples")
    elif model_type == "base":
        train_dataset = train_dataset_real
        print("Using only real data for training")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.train.cnn.batch_size, 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.train.cnn.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    return train_loader, val_loader, class_to_idx
