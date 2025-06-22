from pathlib import Path
import torch
import numpy as np
from omegaconf import OmegaConf
import os
from lightning import Trainer
from .model import VAE
from .dataset import VAEDataModule
from lightning.pytorch.loggers import WandbLogger
import wandb
from lightning.pytorch.callbacks import RichProgressBar

def vae_train_model():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    params_path = PROJECT_ROOT / "params.yaml"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    torch.manual_seed(123)
    np.random.seed(123)

    cfg = OmegaConf.load(params_path)
    vae_params = cfg.train.vae

    image_size = int(vae_params.image_size)
    batch_size = int(vae_params.batch_size)
    num_epochs = int(vae_params.num_epoch)
    learning_rate = float(vae_params.lr)
    latent_dim = int(vae_params.latent_dim)
    beta = float(vae_params.beta)
    kl_anneal_epochs = int(vae_params.kl_anneal_epochs)
    use_perceptual = vae_params.use_perceptual_loss
    log = vae_params.log
    
    
    checkpoint_path = Path(vae_params.checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    final_model = Path(vae_params.final_model)

    datamodule = VAEDataModule(
        train_dir=cfg.data.train_mel,
        val_dir=cfg.data.test_mel,  # Lub stwórz osobny val set
        batch_size=batch_size,
        image_size=image_size
    )


    if log:
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            raise ValueError("No W&B API Key exported!")
        logger = WandbLogger(
            project=cfg.wandb.project_name,
            name="vae_training",
            tags=["bce-loss", "perceptual", "increasing-beta", "kl-div"],
            group='vae',
            config={
                "image_size": image_size,
                "batch_size": batch_size,
                "num_epoch": num_epochs,
                "lr": learning_rate,
                "beta": beta,
                "latent_dim": latent_dim,
                "perceptual_loss": use_perceptual,
                "kl_anneal_epochs": kl_anneal_epochs
            }
        )
        
 
    model = VAE(latent_dim=latent_dim, lr=learning_rate, beta=beta, use_perceptual=use_perceptual, kl_epochs=kl_anneal_epochs)
    trainer = Trainer(
        max_epochs=num_epochs,
        accelerator=device,
        logger=logger,
        default_root_dir=checkpoint_path,
        callbacks=[RichProgressBar()],
    )
    
    trainer.fit(model, datamodule)
    
    if log:
        if logger is not None:
            wandb.finish()
    model_dir = final_model.parent
    model_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), final_model)


