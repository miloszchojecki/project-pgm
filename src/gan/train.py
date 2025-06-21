import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from .models import Generator, Discriminator, ADA
from pathlib import Path
from utils.dataset import SingleClassDataset
from utils.transform import create_transformation
from utils.utils import initialize_wandb
import wandb
from omegaconf import OmegaConf


def train_gan_model():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    params_path = PROJECT_ROOT / "params.yaml"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    torch.manual_seed(123)
    np.random.seed(123)

    # Hyperparameters
    cfg = OmegaConf.load(params_path)

    gan_params = cfg.train.gan

    z_dim = int(gan_params.z_dim)
    image_size = int(gan_params.image_size)
    batch_size = int(gan_params.batch_size)
    num_epochs = int(gan_params.num_epoch)
    learning_rate = float(gan_params.lr)
    log = gan_params.log
    checkpoint_path = Path(gan_params.checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    final_model = Path(gan_params.final_model)

    if log:
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            raise ValueError("No W&B API Key exported!")
        run = initialize_wandb(
            api_key=api_key,
            project_name=cfg.wandb.project_name,
            exp_name="gan_training",
            group='gan',
            config={
                "image_size": image_size,
                "batch_size": batch_size,
                "num_epoch": num_epochs,
                "z_dim": z_dim,
                "lr": learning_rate,
            },
        )

    data_dir = cfg.data.train_mel
    transform = create_transformation(image_size)
    try:
        dataset = SingleClassDataset(folder=data_dir, transform=transform)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        print(f"Dataset size: {len(dataset)} images")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

    # Models
    G = Generator(z_dim=z_dim).to(device)
    D = Discriminator().to(device)
    ada = ADA()

    def print_model_summary(model, model_name):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{model_name}")
        print(f"  Total parameters: {total_params}")
        print(f"  Trainable parameters: {trainable_params}")

    print_model_summary(G, "Generator")
    print_model_summary(D, "Discriminator")

    # Optimizers
    g_opt = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(0.0, 0.99))
    d_opt = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(0.0, 0.99))

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        G.train()
        D.train()

        d_losses = []
        g_losses = []
        r1_penalties = []

        pbar = tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", total=len(dataloader)
        )
        for i, real_images in enumerate(pbar):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            print(f"Iteration {i}, Batch size: {batch_size}")  # Debug print
            z = torch.randn(batch_size, z_dim).to(device)

            # === Train Discriminator ===
            with torch.no_grad():
                fake_images = G(z)

            real_images_aug = ada.apply(real_images)
            real_images_aug.requires_grad_(True)
            d_real = D(real_images_aug)
            d_fake = D(fake_images.detach())

            real_labels = torch.ones_like(d_real)
            fake_labels = torch.zeros_like(d_fake)

            d_loss_real = criterion(d_real, real_labels)
            d_loss_fake = criterion(d_fake, fake_labels)
            gamma = 10.0
            grad = torch.autograd.grad(
                d_real.sum(), real_images_aug, create_graph=True
            )[0]
            r1_penalty = gamma * 0.5 * (grad.norm(2, dim=[1, 2, 3]) ** 2).mean()
            d_loss = d_loss_real + d_loss_fake + r1_penalty

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            ada.update(d_real)

            # === Train Generator ===
            z = torch.randn(batch_size, z_dim).to(device)
            fake_images = G(z)
            d_fake = D(fake_images)
            g_loss = criterion(d_fake, real_labels)

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            pbar.set_postfix(
                {
                    "D Loss": d_loss.item(),
                    "G Loss": g_loss.item(),
                    "R1 Penalty": r1_penalty.item(),
                }
            )

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            r1_penalties.append(r1_penalty.item())
        
        if log:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "avg_d_loss": np.mean(d_losses),
                    "avg_g_loss": np.mean(g_losses),
                    "avg_r1_penalty": np.mean(r1_penalties),
                }
            )
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            try:
                torch.save(
                    {
                        "epoch": epoch,
                        "generator_state_dict": G.state_dict(),
                        "discriminator_state_dict": D.state_dict(),
                        "g_opt_state_dict": g_opt.state_dict(),
                        "d_opt_state_dict": d_opt.state_dict(),
                    },
                    Path(checkpoint_path, f"checkpoint_epoch_{epoch + 1}.pth"),
                )
                print(f"Checkpoint saved at epoch {epoch + 1}")
            except Exception as e:
                print(f"Failed to save checkpoint: {e}")
    if log:
        if run is not None:
            wandb.finish()

    # Save final generator
    try:
        torch.save(
            {
                "generator_state_dict": G.state_dict(),
                "discriminator_state_dict": D.state_dict(),
                "generator_config": {
                    "z_dim": z_dim,
                    "image_size": image_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                },
            },
            final_model,
        )
        print(f"Final model saved: {final_model}")
    except Exception as e:
        print(f"Failed to save final model: {e}")
    return G


if __name__ == "__main__":
    train_gan_model()