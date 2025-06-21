import argparse
import torch
from torch.utils.data import DataLoader
from model import ClassicCNN
from dataset import get_dataloaders
from metrics import compute_metrics
from omegaconf import OmegaConf
from pathlib import Path

def validate_model(model, val_loader, device):
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    params_path = PROJECT_ROOT / "params.yaml"

    parser = argparse.ArgumentParser(description="Evaluate ClassicCNN model")
    parser.add_argument("--model", type=str, default="base", choices=["base", "gan", "diffusion", "vae"])
    args = parser.parse_args()

    cfg = OmegaConf.load(params_path)

    _, val_loader, _ = get_dataloaders(cfg, args.model)

    num_classes = int(cfg.train.cnn.num_classes)
    model_path = cfg.train.cnn.final_model[args.model]

    model = ClassicCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))

    val_acc, labels, preds = validate_model(model, val_loader, device)
    metrics = compute_metrics(labels, preds)

    print(f"[{args.model}] Validation Metrics:")
    print(f"  Accuracy: {val_acc:.2f}%")
    for metric, value in metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")

if __name__ == "__main__":
    main()