import os
import torch
import torchvision.utils as vutils
from tqdm import tqdm

from models import Generator  # import your GAN Generator here


def load_generator(model_path, device='cpu', z_dim=256):
    """Load trained GAN generator model."""
    print(f"Loading generator model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Use config from checkpoint if available, else fallback to passed z_dim
    cfg = checkpoint.get('generator_config', {})
    z_dim = cfg.get('z_dim', z_dim)

    model = Generator(z_dim=z_dim).to(device)
    model.load_state_dict(checkpoint['generator_state_dict'])
    model.eval()
    print("Generator model loaded successfully")
    return model


def generate_samples(generator, num_samples=64, z_dim=256, device='cpu', save_dir='results/gan/samples', base_dir="/", truncation_psi=1.0):
    print(f"Generating {num_samples} samples on {device}...")
    save_path = os.path.join(base_dir, save_dir)
    os.makedirs(save_path, exist_ok=True)

    generator.eval()
    with torch.no_grad():
        batch_size = 16
        all_samples = []
        for i in tqdm(range(0, num_samples, batch_size)):
            current_batch_size = min(batch_size, num_samples - i)
            z = torch.randn(current_batch_size, z_dim, device=device)
            samples = generator(z, truncation_psi=truncation_psi)
            samples = (samples + 1) / 2
            all_samples.append(samples.cpu())
        
        all_samples = torch.cat(all_samples, dim=0)
        
        for i, sample in enumerate(all_samples):
            vutils.save_image(sample, os.path.join(save_path, f'sample_{i:03d}.png'))

    print(f"Samples saved to: {save_path}")
    return all_samples


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    base_dir = '/home/soltys/sztuczna_inteligencja/sem1/pgm/project-pgm'
    model_path = os.path.join(base_dir, 'results', 'gan', 'generator_final.pth')
    z_dim = 256  # default latent dim, update if your model uses different
    generator = load_generator(model_path, device=device, z_dim=z_dim)

    samples = generate_samples(generator, num_samples=64, z_dim=z_dim, device=device, base_dir=base_dir, truncation_psi=1.0)
    print("Sample generation completed!")
