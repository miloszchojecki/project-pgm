import lightning as L
import torch.nn as nn
import torch
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),              
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),             
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),         
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)
        
    def forward(self, z):
        x = self.conv_layers(z)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    
class Decoder(nn.Module):
    def __init__(self, latent_dim=128, output_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 16 * 16)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),   
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),    
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),     
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, 4, 2, 1),  
            nn.Tanh()
        )
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 16, 16)
        x = self.deconv_layers(x)
        return x
        

class VAE(L.LightningModule):
    def __init__(self, input_channels=3, latent_dim=128, lr=1e-4, beta=0.3):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(latent_dim, input_channels)
        
        self.lr = lr

    def reparameterize(self, mu, logvar):
        std = torch.sqrt(torch.exp(logvar))
        noise = torch.randn_like(mu)
        return mu + noise * std

    def train_loss(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss, kl_loss, recon_loss + self.hparams.beta * kl_loss
    
    def training_step(self, batch, batch_idx):
        x = batch
        recon_x, mu, logvar = self(x)
        recon_loss, kl_loss, loss = self.train_loss(recon_x, x, mu, logvar)
        self.log("train/total_loss", loss)
        self.log("train/kl_loss", kl_loss)
        self.log("train/recon_loss", recon_loss)
        return loss
        
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        
        return recon_x, mu, logvar
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def generate(self, num_samples=16, device='cuda'):
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.hparams.latent_dim).to(device)
            samples = self.decoder(z)
        return samples