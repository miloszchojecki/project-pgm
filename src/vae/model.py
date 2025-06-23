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
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 256, 256)
            dummy_output = self.conv_layers(dummy_input)
            self.feature_size = dummy_output.view(1, -1).size(1)
        
        self.fc_mu = nn.Linear(self.feature_size, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_size, latent_dim)

    def forward(self, z):
        x = self.conv_layers(z)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=128, output_channels=3, feature_size=32768):
        super().__init__()
        self.conv_h, self.conv_w, self.conv_c = 8, 8, 512
        self.feature_size = feature_size
        
        self.fc = nn.Linear(latent_dim, self.feature_size)
        
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  
            nn.BatchNorm2d(256),
            nn.ReLU(),
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
        x = x.view(x.size(0), self.conv_c, self.conv_h, self.conv_w)
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
        image_size=256,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(latent_dim, input_channels, self.encoder.feature_size)
        self.lr = lr
        self.beta = beta
        self.kl_epochs = kl_epochs
        self.use_perceptual = use_perceptual
        self.kl_weight = 0.0
        self.reconstruction = nn.MSELoss(reduction='mean')

        if use_perceptual:
            vgg_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self.vgg = nn.Sequential(*list(vgg_model.features.children())[:16])
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
        logvar = torch.clamp(logvar, min=-20, max=20)
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        return mu + noise * std

    def train_loss(self, recon_x, x, mu, logvar):
        kl_per_dim = 0.5 * (-1 - logvar + mu.pow(2) + logvar.exp())
        free_bits = 0.2
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
        
        kl = torch.mean(torch.sum(kl_per_dim, dim=1))
        kl_loss = self.beta * self.kl_weight * kl
        recon_loss = self.reconstruction(recon_x, x)
        total_loss = recon_loss + kl_loss
        
        perceptual_loss = torch.tensor(0.0, device=x.device)
        if self.use_perceptual:
            perceptual_loss = 0.5 * self.perceptual_loss(recon_x, x)
            total_loss += perceptual_loss
            return recon_loss, perceptual_loss, kl_loss, total_loss
        else:
            return recon_loss, kl_loss, total_loss

    def training_step(self, batch, batch_idx):
        x = batch
        
        if self.current_epoch < self.kl_epochs:
            progress = max(0, (self.current_epoch - self.kl_epochs/2) / (self.kl_epochs/2))
            self.kl_weight = min(0.8, progress)
        else:
            self.kl_weight = 0.8
        
        recon_x, mu, logvar = self(x)
        loss_result = self.train_loss(recon_x, x, mu, logvar)
        
        if self.use_perceptual and len(loss_result) == 4:
            recon_loss, perceptual_loss, kl_loss, loss = loss_result
            self.log("train/perceptual_loss", perceptual_loss, prog_bar=False,)
        else:
            recon_loss, kl_loss, loss = loss_result[:3]
        self.log("train/total_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train/kl_loss", kl_loss, prog_bar=True,)
        self.log("train/recon_loss", recon_loss, prog_bar=False,)
        self.log("kl_weight", self.kl_weight)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        recon_x, mu, logvar = self(x)
        loss_result = self.train_loss(recon_x, x, mu, logvar)
        
        if self.use_perceptual and len(loss_result) == 4:
            recon_loss, perceptual_loss, kl_loss, loss = loss_result
            self.log("val/perceptual_loss", perceptual_loss)
        else:
            recon_loss, kl_loss, loss = loss_result[:3]
            
        self.log("val/total_loss", loss, prog_bar=True)
        self.log("val/kl_loss", kl_loss)
        self.log("val/recon_loss", recon_loss)
        return loss

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)

        return recon_x, mu, logvar

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'gradient_clip_val': 1.0,
            'gradient_clip_algorithm': 'norm'
        }

    def generate(self, num_samples=16, device="cuda"):
        self.eval()
        with torch.no_grad():
            latent_dim = self.encoder.fc_mu.out_features
            z = torch.randn(num_samples, latent_dim).to(device)
            samples = self.decoder(z)
        return samples
