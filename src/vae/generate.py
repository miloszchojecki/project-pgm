import torch
import torchvision.utils as vutils
from pathlib import Path
from omegaconf import OmegaConf
from .model import VAE

def generate_vae_samples():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    cfg = OmegaConf.load(PROJECT_ROOT / "params.yaml")
    
    
    final_model = Path(cfg.train.vae.final_model)
    result_path = Path(cfg.generate.vae.result_path)
    num_samples = int(cfg.generate.num_samples)
    latent_dim = int(cfg.train.vae.latent_dim)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = VAE(
        latent_dim=int(cfg.train.vae.latent_dim),
        lr=float(cfg.train.vae.lr),
        beta = float(cfg.train.vae.beta),
        use_perceptual = bool(cfg.train.vae.use_perceptual_loss)
    )
    
    # Load weights
    model.load_state_dict(torch.load(final_model, map_location=device))
    model = model.to(device)
    model.eval()
    
    result_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} samples...")
    
    # Generate samples in batches
    batch_size = 16
    sample_idx = 0
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            
            generated_images = model.generate(num_samples=current_batch_size, device=device)
            generated_images = (generated_images + 1) / 2

            for j in range(current_batch_size):
                img_path = result_path / f"generated_{sample_idx:04d}.png"
                vutils.save_image(generated_images[j], img_path)
                sample_idx += 1

if __name__ == "__main__":
    generate_vae_samples()