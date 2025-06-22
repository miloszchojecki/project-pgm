import lightning as L
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models


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
            # nn.Tanh(),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 16, 16)
        x = self.deconv_layers(x)
        return x


class VAE(L.LightningModule):
    def __init__(
        self,
        input_channels=3,
        latent_dim=128,
        lr=1e-4,
        beta=0.3,
        kl_epochs=10,
        use_perceptual=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(latent_dim, input_channels)
        self.lr = lr

        # self.reconstruction = nn.MSELoss(reduction='mean')
        self.reconstruction = nn.BCELoss(reduction='mean')

        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[
            :16
        ]
        for params in self.vgg.parameters():
            params.requires_grad = False
        self.vgg.eval()

    def normalize_for_vgg(self, x):
        if x.min() < 0:
            x = (x + 1) / 2
        mean = torch.tensor([0.485, 0.456, 0.406]).to(x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def perceptual_loss(self, recon_x, x):
        x_norm = self.normalize_for_vgg(x)
        recon_x_norm = self.normalize_for_vgg(recon_x)

        original = self.vgg(x_norm)
        recon = self.vgg(recon_x_norm)

        return F.mse_loss(recon, original)

    def reparameterize(self, mu, logvar):
        std = torch.sqrt(torch.exp(logvar))
        noise = torch.randn_like(mu)
        return mu + noise * std

    def train_loss(self, recon_x, x, mu, logvar):
        # kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    
        # # Free bits: wymuszaj minimum 1.0 nat per wymiar
        # free_bits = 0.1
        # kl_per_dim = torch.max(kl_per_dim, torch.tensor(free_bits).to(x.device))
    
        # kl_loss = self.hparams.beta * kl_per_dim.sum(dim=1).mean()
        kl = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) -1 - logvar, dim=1).mean()
        kl_loss = self.hparams.beta * kl
        x_target = (x + 1) / 2
        batch_size = x.shape[0]
        recon_x = recon_x.view(batch_size, -1)
        x_target = x_target.view(batch_size, -1)
        recon_loss = self.reconstruction(recon_x, x_target)
        total_loss = recon_loss + kl_loss
        
        perceptual_loss = torch.tensor(0.0, device=x.device)
        if self.hparams.use_perceptual:
            perceptual_loss = 0.5 * self.perceptual_loss(recon_x, x)
            total_loss += perceptual_loss
            return recon_loss, perceptual_loss, kl_loss, total_loss
        else:
            return recon_loss, kl_loss, total_loss

    def training_step(self, batch, batch_idx):
        x = batch
        recon_x, mu, logvar = self(x)
        if self.hparams.use_perceptual:
            recon_loss, perceptual_loss, kl_loss, loss = self.train_loss(
                recon_x, x, mu, logvar
            )
            self.log("train/perceptual_loss", perceptual_loss, prog_bar=False,)
        else:
            recon_loss, kl_loss, loss = self.train_loss(recon_x, x, mu, logvar)
        self.log("train/total_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train/kl_loss", kl_loss, prog_bar=True,)
        self.log("train/recon_loss", recon_loss, prog_bar=False,)
        return loss

    def on_train_epoch_start(self) -> None:
        kl_anneal_epochs = self.hparams.get("kl_epochs", 0)
        
        if self.current_epoch > kl_anneal_epochs:
            self.hparams.beta = 0.5

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)

        return recon_x, mu, logvar

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def generate(self, num_samples=16, device="cuda"):
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.hparams.latent_dim).to(device)
            samples = self.decoder(z)
        return samples
