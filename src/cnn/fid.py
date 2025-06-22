import argparse
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from .model import ClassicCNN
from .metrics import compute_fid
import numpy as np
import torch.nn as nn
from pathlib import Path
from ..utils.dataset import FidClassDataset
from ..utils.utils import initialize_wandb
import wandb
import os

def extract_features(model: nn.Module, dataloader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    features = []
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            feats = model(images, return_features=True)
            features.append(feats.cpu())
    return torch.cat(features, dim=0).numpy()

def fid():
    parser = argparse.ArgumentParser(description="Compute FID score")
    parser.add_argument('--model', type=str, required=True, help="Model used for generating images", choices=["gan", "diffusion", "vae"])
    args = parser.parse_args()
    
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    params_path = PROJECT_ROOT / "params.yaml"
    
    cfg = OmegaConf.load(params_path)
    image_size = int(cfg.train.cnn.image_size)
    batch_size = int(cfg.train.cnn.batch_size)
    num_classes = int(cfg.train.cnn.num_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log = cfg.train.cnn.log
    if log:
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            raise ValueError("No W&B API Key exported!")
        run = initialize_wandb(
            api_key=api_key,
            project_name=cfg.wandb.project_name,
            exp_name=f"fid_{args.model}",
            group='fid',
            config={
                "image_size": image_size
            },
        )
        
    real_dataset = FidClassDataset(str(cfg.data.test_mel), image_size)
    gen_dataset = FidClassDataset(str(cfg.generate[args.model].result_path), image_size)

    print("Real data path:", cfg.data.test_mel)
    print("Generated data path:", cfg.generate[args.model].result_path)

    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    gen_loader = DataLoader(gen_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("Real dataset size:", len(real_dataset))
    print("Generated dataset size:", len(gen_dataset))

    model = ClassicCNN(num_classes=num_classes)
    
    model.load_state_dict(torch.load(str(cfg.train.cnn.final_model.base), map_location=device))
    print("Model loaded from:", cfg.train.cnn.final_model.base)
    model.to(device)
    
    features_real = extract_features(model, real_loader, device)
    features_gen = extract_features(model, gen_loader, device)
    fid_score = compute_fid(features_real, features_gen)
    
    print(f"\n FID score (using ClassicCNN features): {fid_score:.4f}")
    if log:
        wandb.log({"fid_score": fid_score})
        run.finish()

if __name__ == "__main__":
    fid()
