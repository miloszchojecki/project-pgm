import argparse
import torch
from torch.utils.data import DataLoader
from config import Config
from model import ClassicCNN
from dataset import CustomImageDataset, get_transforms
from metrics import compute_fid
import numpy as np
import torch.nn as nn

def extract_features(model: nn.Module, dataloader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    features = []
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            feats = model(images, return_features=True)
            features.append(feats.cpu())
    return torch.cat(features, dim=0).numpy()

def main():
    parser = argparse.ArgumentParser(description="Compute FID score")
    parser.add_argument('--model', type=str, required=True, help="Model used for generating images")
    args = parser.parse_args()
    
    config = Config()
    transform = get_transforms(config.image_size)
    
    real_dataset = CustomImageDataset(str(config.dataset_paths["val"] / "mel"), config.image_size)
    gen_dataset = CustomImageDataset(str(config.get_gen_path(args.model)), config.image_size)
    
    real_loader = DataLoader(real_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    gen_loader = DataLoader(gen_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    model = ClassicCNN(num_classes=config.num_classes)
    model.load_state_dict(torch.load(str(config.get_model_path("base")), map_location=config.device))
    model.to(config.device)
    
    features_real = extract_features(model, real_loader, config.device)
    features_gen = extract_features(model, gen_loader, config.device)
    fid_score = compute_fid(features_real, features_gen)
    
    print(f"\n FID score (using ClassicCNN features): {fid_score:.4f}")

if __name__ == "__main__":
    main()
