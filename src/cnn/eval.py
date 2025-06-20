import argparse
import torch
from torch.utils.data import DataLoader
from config import Config
from model import ClassicCNN
from dataset import get_transforms
from metrics import compute_metrics
from torchvision import datasets
from typing import Tuple, List
from torch import nn

def validate_model(model: nn.Module, val_loader: DataLoader, device: str) -> Tuple[float, List[int], List[int]]:
    model.eval()
    correct, total = 0, 0
    all_labels, all_preds = [], []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    return 100 * correct / total, all_labels, all_preds

def main():
    parser = argparse.ArgumentParser(description="Evaluate ClassicCNN model")
    parser.add_argument("--model", type=str, default="base", choices=["base", "gan", "diffusion", "vae"])
    args = parser.parse_args()
    
    config = Config()
    transform = get_transforms(config.image_size)
    
    val_dataset = datasets.ImageFolder(root=str(config.dataset_paths["val"]), transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    model = ClassicCNN(num_classes=config.num_classes).to(config.device)
    model.load_state_dict(torch.load(str(config.get_model_path(args.model)), map_location=config.device))
    
    _, labels, preds = validate_model(model, val_loader, config.device)
    metrics = compute_metrics(labels, preds)
    
    print(f"[{args.model}] Validation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")

if __name__ == "__main__":
    main()