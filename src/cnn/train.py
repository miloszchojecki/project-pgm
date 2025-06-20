import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import Config
from model import ClassicCNN
from dataset import get_dataloaders
from torch.utils.data import DataLoader
from typing import Tuple
from eval import validate_model

def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: str, model_type: str, epoch: int) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    progress_bar = tqdm(train_loader, desc=f"[{model_type}] Epoch {epoch+1}", leave=False)
    
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
        progress_bar.set_postfix(loss=total_loss/total, acc=f"{acc:.2f}%")
    
    return total_loss/total, acc

def main():
    parser = argparse.ArgumentParser(description="Train ClassicCNN model")
    parser.add_argument("--model", type=str, default="base", choices=["base", "gan", "diffusion", "vae"])
    args = parser.parse_args()
    
    config = Config()
    train_loader, val_loader, class_to_idx = get_dataloaders(config, args.model)
    
    model = ClassicCNN(num_classes=config.num_classes).to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    for epoch in range(config.epochs):
        train_loss, train_acc = train_model(
            model, train_loader, criterion, optimizer, config.device, args.model, epoch
        )
        print(f"[{args.model}] Epoch {epoch+1} complete: Loss = {train_loss:.4f}, Accuracy = {train_acc:.2f}%")
    
    val_acc, labels, preds = validate_model(model, val_loader, config.device)
    print(f"[{args.model}] Validation Accuracy: {val_acc:.2f}%")
    
    torch.save(model.state_dict(), str(config.get_model_path(args.model)))
    print(f"Model saved to {config.get_model_path(args.model)}")

if __name__ == "__main__":
    main()
