import argparse
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import ClassicCNN
from dataset import get_dataloaders
from torch.utils.data import DataLoader
from typing import Tuple
from eval import validate_model
from pathlib import Path
from utils.utils import initialize_wandb
import os
import wandb


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    model_type: str,
    epoch: int,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    progress_bar = tqdm(
        train_loader, desc=f"[{model_type}] Epoch {epoch + 1}", leave=False
    )

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

        acc = 100 * correct / total
        progress_bar.set_postfix(loss=total_loss / total, acc=f"{acc:.2f}%")

    return total_loss / total, acc


def main():
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise ValueError("No W&B API Key exported!")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    params_path = PROJECT_ROOT / "params.yaml"

    parser = argparse.ArgumentParser(description="Train ClassicCNN model")
    parser.add_argument(
        "--model", type=str, default="base", choices=["base", "gan", "diffusion", "vae"]
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(params_path)

    train_loader, val_loader, class_to_idx = get_dataloaders(cfg, args.model)

    num_classes = int(cfg.train.cnn.num_classes)
    image_size = int(cfg.train.cnn.image_size)
    batch_size = int(cfg.train.cnn.batch_size)
    learning_rate = float(cfg.train.cnn.lr)
    num_epochs = int(cfg.train.cnn.num_epochs)

    model_path = cfg.train.cnn.final_model[args.model]

    model = ClassicCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    run = initialize_wandb(
        api_key=api_key,
        project_name=cfg.wandb.project_name,
        exp_name="diffusion_training",
        group='cnn',
        config={
            "image_size": image_size,
            "batch_size": batch_size,
            "num_epoch": num_epochs,
            "lr": learning_rate,
        },
    )

    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(
            model, train_loader, criterion, optimizer, device, args.model, epoch
        )
        print(
            f"[{args.model}] Epoch {epoch + 1} complete: Loss = {train_loss:.4f}, Accuracy = {train_acc:.2f}%"
        )
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
            }
        )

    val_acc, labels, preds = validate_model(model, val_loader, device)
    print(f"[{args.model}] Validation Accuracy: {val_acc:.2f}%")
    wandb.log(
        {
            "epoch": num_epochs,
            "val_acc": val_acc,
        }
    )

    if run is not None:
        wandb.finish()
        
    torch.save(model.state_dict(), str(model_path))
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
