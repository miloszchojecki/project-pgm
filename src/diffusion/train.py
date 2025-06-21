import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models import LargeConvDenoiserNetwork
from utils.transform import create_transformation
from pathlib import Path
from utils.utils import initialize_wandb
from utils.dataset import SingleClassDataset
import wandb
from omegaconf import OmegaConf


def apply(coefficients: np.array, timesteps: torch.tensor, x: torch.tensor):
    """
    Apply coefficients to tensor x based on timesteps.
    """
    factors = (
        torch.from_numpy(coefficients).to(device=timesteps.device)[timesteps].float()
    )
    K = x.dim() - 1
    factors = factors.view(-1, *([1] * K)).expand(x.shape)
    return factors * x


class GaussianDiffusion:
    """
    Gaussian Diffusion implementation for image generation.
    """

    def __init__(self, num_timesteps=1000):
        self.num_timesteps = num_timesteps
        self.timesteps = list(range(self.num_timesteps))[::-1]

        self.betas = np.linspace(start=0.0001, stop=0.02, num=self.num_timesteps)
        self.setup_noise_scheduler(self.betas)

    def setup_noise_scheduler(self, betas):
        self.alphas = 1.0 - betas
        self.alpha_bars = np.cumprod(self.alphas, axis=0)
        self.alpha_bars_prev = np.append(1.0, self.alpha_bars[:-1])

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion process - add noise to clean image."""
        if noise is None:
            noise = torch.randn_like(x_0)

        mean = apply(np.sqrt(self.alpha_bars), t, x_0)
        std_dev = apply(np.sqrt(1.0 - self.alpha_bars), t, noise)
        return mean + std_dev

    def q_posterior(self, x_t, x_0, t, noise=None):
        """Posterior of forward process."""
        if noise is None:
            noise = torch.randn_like(x_0)

        term1_coeff = apply(
            np.sqrt(self.alphas)
            * (1.0 - self.alpha_bars_prev)
            / (1.0 - self.alpha_bars),
            t,
            x_t,
        )
        term2_coeff = apply(
            np.sqrt(self.alpha_bars_prev) * self.betas / (1.0 - self.alpha_bars), t, x_0
        )
        posterior_mean = term1_coeff + term2_coeff

        posterior_variance = apply(
            (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars) * self.betas,
            t,
            torch.ones_like(x_0),
        )
        posterior_std_dev = torch.sqrt(posterior_variance)

        return posterior_mean + posterior_std_dev * noise

    def _predict_x_0_from_eps(self, x_t, t, eps):
        """Predict x_0 from noise prediction."""
        return apply(np.sqrt(1.0 / self.alpha_bars), t, x_t) - apply(
            np.sqrt(1.0 / self.alpha_bars - 1.0), t, eps
        )

    def train_losses(self, model, x_0):
        """Training loss function."""
        t = torch.randint(
            0, self.num_timesteps, (x_0.shape[0],), device=x_0.device
        ).long()
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = model(x_t, t)
        loss = nn.functional.mse_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def p_sample_loop(
        self, model, noise, num_inference_steps=1000, clip=True, quiet=False
    ):
        """Reverse diffusion process - generate samples from noise."""
        self._respace(num_timesteps=num_inference_steps)

        x_t = noise
        bsz = x_t.shape[0]

        pbar = (
            tqdm(enumerate(self.timesteps), desc="Generating", total=self.num_timesteps)
            if not quiet
            else enumerate(self.timesteps)
        )

        for idx, time in pbar:
            t = torch.tensor([time] * bsz, device=x_t.device).long()
            i = torch.tensor(
                [self.num_timesteps - idx - 1] * bsz, device=x_t.device
            ).long()

            eps = model(x_t, t)
            x_0 = self._predict_x_0_from_eps(x_t, i, eps)

            if clip:
                x_0 = x_0.clamp(-1, 1)

            x_t = self.q_posterior(x_t, x_0, i)

        self._respace(1000)
        return x_0.cpu().numpy()

    def _respace(self, num_timesteps):
        """Change number of timesteps for inference."""
        betas = np.linspace(start=0.0001, stop=0.02, num=1000)
        self.setup_noise_scheduler(betas)

        self.num_timesteps = num_timesteps
        self.timesteps = np.linspace(
            999, 0, self.num_timesteps, dtype=int, endpoint=True
        )

        last_alpha_cumprod = 1.0
        self.betas = []

        for i, alpha_bar in enumerate(self.alpha_bars):
            if i in self.timesteps:
                self.betas.append(1 - alpha_bar / last_alpha_cumprod)
                last_alpha_cumprod = alpha_bar

        self.betas = np.array(self.betas)
        self.setup_noise_scheduler(self.betas)


def train_diffusion_model(api_key):
    """Main training function."""
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    params_path = PROJECT_ROOT / "params.yaml"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    torch.manual_seed(123)
    np.random.seed(123)


    cfg = OmegaConf.load(params_path)
    diff_params = cfg.train.diffusion

    image_size = int(diff_params.image_size)
    batch_size = int(diff_params.batch_size)
    num_epochs = int(diff_params.num_epoch)
    learning_rate = float(diff_params.lr)

    checkpoint_path = Path(diff_params.checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    run = initialize_wandb(
        api_key=api_key,
        project_name=cfg.wandb.project_name,
        exp_name="diffusion_training",
        group='diffusion',
        config={
            "image_size": image_size,
            "batch_size": batch_size,
            "num_epoch": num_epochs,
            "lr": learning_rate,
        },
    )

    final_model = Path(diff_params.final_model)

    data_dir = cfg.data.train_mel
    transform = create_transformation(image_size)
    dataset = SingleClassDataset(folder=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    print(f"Dataset size: {len(dataset)} images")

    diffusion = GaussianDiffusion(num_timesteps=1000)
    model = LargeConvDenoiserNetwork(
        in_channels=3,
        out_channels=3,
        channels=[64, 128, 256, 512],
        layers_per_block=2,
        add_attention=True,
        attention_head_dim=64,
        global_skip_connection=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    losses = []

    for epoch in range(num_epochs):
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, x_0 in enumerate(pbar):
            x_0 = x_0.to(device)

            optimizer.zero_grad()
            loss = diffusion.train_losses(model, x_0)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        wandb.log({"epoch": epoch + 1, "avg_loss": avg_loss})

        if (epoch + 1) % 10 == 0:
            checkpoint = Path(checkpoint_path, f"_epoch_{epoch + 1}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                checkpoint,
            )
            print(f"Checkpoint saved: {checkpoint}")

    if run is not None:
        wandb.finish()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "diffusion_config": {
                "num_timesteps": diffusion.num_timesteps,
                "betas": diffusion.betas,
            },
        },
        final_model,
    )
    print(f"Final model saved: {final_model}")

    return model, diffusion

def train_diffusion():
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise ValueError("No W&B API Key exported!")
    model, diffusion = train_diffusion_model(api_key)
    print("Training and generation completed!")


if __name__ == "__main__":
    main()
    
