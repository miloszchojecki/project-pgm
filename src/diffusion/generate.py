import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.utils as vutils

from models import LargeConvDenoiserNetwork


def apply(coefficients: np.array, timesteps: torch.tensor, x: torch.tensor):
    """Apply coefficients to tensor x based on timesteps."""
    factors = torch.from_numpy(coefficients).to(device=timesteps.device)[timesteps].float() 
    K = x.dim() - 1
    factors = factors.view(-1, *([1]*K)).expand(x.shape)
    return factors * x


class GaussianDiffusion:
    """Gaussian Diffusion implementation for image generation."""
    def __init__(self, num_timesteps=1000):
        self.num_timesteps = num_timesteps
        self.timesteps = list(range(self.num_timesteps))[::-1]

        self.betas = np.linspace(start=0.0001, stop=0.02, num=self.num_timesteps)
        self.setup_noise_scheduler(self.betas)

    def setup_noise_scheduler(self, betas):
        self.alphas = 1.0 - betas
        self.alpha_bars = np.cumprod(self.alphas, axis=0)
        self.alpha_bars_prev = np.append(1.0, self.alpha_bars[:-1])

    def _predict_x_0_from_eps(self, x_t, t, eps):
        """Predict x_0 from noise prediction."""
        return (
            apply(np.sqrt(1.0 / self.alpha_bars), t, x_t) - 
            apply(np.sqrt(1.0 / self.alpha_bars - 1.0), t, eps)
        )

    def q_posterior(self, x_t, x_0, t, noise=None):
        """Posterior of forward process."""
        if noise is None:
            noise = torch.randn_like(x_0)

        term1_coeff = apply(np.sqrt(self.alphas) * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars), t, x_t)
        term2_coeff = apply(np.sqrt(self.alpha_bars_prev) * self.betas / (1.0 - self.alpha_bars), t, x_0)
        posterior_mean = term1_coeff + term2_coeff
        
        posterior_variance = apply((1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars) * self.betas, t, torch.ones_like(x_0))
        posterior_std_dev = torch.sqrt(posterior_variance)
        
        return posterior_mean + posterior_std_dev * noise

    @torch.no_grad()
    def p_sample_loop(self, model, noise, num_inference_steps=100, clip=True, quiet=False):
        """Reverse diffusion process - generate samples from noise."""
        self._respace(num_timesteps=num_inference_steps)

        x_t = noise
        bsz = x_t.shape[0]

        pbar = tqdm(enumerate(self.timesteps), desc='Generating', total=len(self.timesteps)) if not quiet else enumerate(self.timesteps)

        for idx, time in pbar:
            t = torch.tensor([time] * bsz, device=x_t.device).long()
            i = torch.tensor([len(self.timesteps) - idx - 1] * bsz, device=x_t.device).long()

            eps = model(x_t, t)
            x_0 = self._predict_x_0_from_eps(x_t, i, eps)

            if clip:
                x_0 = x_0.clamp(-1, 1)

            if idx < len(self.timesteps) - 1:  
                x_t = self.q_posterior(x_t, x_0, i)
            else:
                x_t = x_0

        self._respace(1000)
        return x_t.cpu().numpy()
    
    def _respace(self, num_timesteps):
        """Change number of timesteps for inference."""
        betas = np.linspace(start=0.0001, stop=0.02, num=1000)
        self.setup_noise_scheduler(betas) 

        self.num_timesteps = num_timesteps
        self.timesteps = np.linspace(999, 0, self.num_timesteps, dtype=int, endpoint=True)

        last_alpha_cumprod = 1.0
        self.betas = []

        for i, alpha_bar in enumerate(self.alpha_bars):
            if i in self.timesteps:
                self.betas.append(1 - alpha_bar / last_alpha_cumprod)
                last_alpha_cumprod = alpha_bar
        
        self.betas = np.array(self.betas)
        self.setup_noise_scheduler(self.betas)


def load_model(model_path, device='cpu'):
    """Load trained model."""
    print(f"Loading model from: {model_path}")
    
    model = LargeConvDenoiserNetwork(
        in_channels=3,
        out_channels=3,
        channels=[64, 128, 256, 512],
        layers_per_block=2,
        add_attention=True,
        attention_head_dim=64,
        global_skip_connection=True,
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully")
    return model


def generate_samples(model, diffusion, num_samples=64, device='cpu'):
    """Generate new samples using trained model."""
    print(f"Generating {num_samples} samples on {device}...")
    
    model.eval()
    with torch.no_grad():
        noise_shape = (num_samples, 3, 256, 256) 
        noise = torch.randn(noise_shape).to(device)
        
        samples = diffusion.p_sample_loop(model, noise, num_inference_steps=50, clip=True)
        
        samples_tensor = torch.from_numpy(samples)
        samples_tensor = (samples_tensor + 1) / 2  
        samples_tensor = torch.clamp(samples_tensor, 0, 1)  
        
        os.makedirs('/home/milosz/Desktop/project-pgm/results/diffusion/samples', exist_ok=True)
        
        for i, sample in enumerate(samples_tensor):
            vutils.save_image(sample, f'/home/milosz/Desktop/project-pgm/results/diffusion/samples/sample_{i:03d}.png')
        
        print(f"Generated samples saved to: /home/milosz/Desktop/project-pgm/results/diffusion/samples/")
        
        return samples_tensor


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model_path = '/home/milosz/Desktop/project-pgm/results/diffusion/final_model.pth'
    model = load_model(model_path, device)
    
    diffusion = GaussianDiffusion(num_timesteps=1000)
    
    samples = generate_samples(model, diffusion, num_samples=64, device=device)
    
    print("Sample generation completed")
