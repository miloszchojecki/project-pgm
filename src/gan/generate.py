from omegaconf import OmegaConf
import torch
import torchvision.utils as vutils
from tqdm import tqdm
from pathlib import Path
from models import Generator
import yaml


def load_generator(model_path, device="cpu", z_dim=256):
    """Load trained GAN generator model."""
    print(f"Loading generator model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Use config from checkpoint if available, else fallback to passed z_dim
    cfg = checkpoint.get("generator_config", {})
    z_dim = cfg.get("z_dim", z_dim)

    model = Generator(z_dim=z_dim).to(device)
    model.load_state_dict(checkpoint["generator_state_dict"])
    model.eval()
    print("Generator model loaded successfully")
    return model


def generate_samples(
    generator,
    device,
    save_dir,
    num_samples=64,
    z_dim=256,
    truncation_psi=1.0,
):
    print(f"Generating {num_samples} samples on {device}...")
    

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
            vutils.save_image(sample, Path(save_dir, f"sample_{i:03d}.png"))

    print(f"Samples saved to: {save_dir}")
    return all_samples


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    params_path = PROJECT_ROOT / "params.yaml"
    
    with open(params_path, "r") as file:
        all_params = yaml.safe_load(file)
        
    cfg = OmegaConf.load(params_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_path = Path(cfg.train.gan.final_model)
    result_path = Path(cfg.generate.gan.result_path)
    result_path.mkdir(exist_ok=True)
    
    z_dim = int(cfg.train.gan.z_dim)
    generator = load_generator(model_path, device=device, z_dim=z_dim)

    samples = generate_samples(
        generator=generator,
        device=device,
        save_dir=result_path,
        num_samples=64,
        z_dim=z_dim,
        truncation_psi=1.0,
    )
    print("Sample generation completed!")
