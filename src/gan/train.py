import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import Generator, Discriminator, ADA
from dataset import CustomImageDataset

def train_gan_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    # Hyperparameters
    image_size = 256
    z_dim = 256
    batch_size = 64 
    num_epochs = 200
    learning_rate = 2e-4
    checkpoint_dir = '/home/soltys/sztuczna_inteligencja/sem1/pgm/project-pgm/results/gan'

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Dataset
    data_dir = '/home/soltys/sztuczna_inteligencja/sem1/pgm/project-pgm/data'
    try:
        dataset = CustomImageDataset(data_dir, image_size=image_size, train=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        print(f"Dataset size: {len(dataset)} images")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

    # Models
    G = Generator(z_dim=z_dim).to(device)
    D = Discriminator().to(device)
    ada = ADA()

    # Optimizers
    g_opt = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(0.0, 0.99))
    d_opt = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(0.0, 0.99))

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        G.train()
        D.train()

        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', total=len(dataloader))
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
            grad = torch.autograd.grad(d_real.sum(), real_images_aug, create_graph=True)[0]
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

            pbar.set_postfix({'D Loss': d_loss.item(), 'G Loss': g_loss.item(), 'R1 Penalty': r1_penalty.item()})

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            try:
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': G.state_dict(),
                    'discriminator_state_dict': D.state_dict(),
                    'g_opt_state_dict': g_opt.state_dict(),
                    'd_opt_state_dict': d_opt.state_dict(),
                }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
                print(f"Checkpoint saved at epoch {epoch+1}")
            except Exception as e:
                print(f"Failed to save checkpoint: {e}")

    # Save final generator
    final_model_path = os.path.join(checkpoint_dir, 'generator_final.pth')
    try:
        torch.save({
            'generator_state_dict': G.state_dict(),
            'discriminator_state_dict': D.state_dict(),
            'generator_config': {
                'z_dim': z_dim,
                'image_size': image_size,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
            }
        }, final_model_path)
        print(f"Final model saved: {final_model_path}")
    except Exception as e:
        print(f"Failed to save final model: {e}")
    return G

if __name__ == '__main__':
    train_gan_model()