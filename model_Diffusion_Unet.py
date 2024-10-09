import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from get_cifar import get_loaders
from utils import count_parameters


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule for beta (variance) values in diffusion models.

    Args:
        timesteps (int): Number of timesteps (T) in the diffusion process.
        s (float): Small constant added for numerical stability.

    Returns:
        betas (torch.Tensor): A tensor of size (timesteps,) containing the beta values.
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, 0, 0.999)  # To ensure numerical stability
    return torch.tensor(betas, dtype=torch.float32)


class Unet(nn.Module):
    def __init__(self, n_steps=1000):
        super(Unet, self).__init__()
        self.n_steps = n_steps
        #self.beta = torch.linspace(1e-4, 0.02, n_steps)
        self.beta = cosine_beta_schedule(n_steps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        features = 128

        # Encoder path
        self.encoder1 = Unet._block(3, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = Unet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = Unet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = Unet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = Unet._block(features * 8, features * 16, name="bottleneck")

        # Decoder path
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = Unet._block((features * 8) * 2, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = Unet._block((features * 4) * 2, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = Unet._block((features * 2) * 2, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = Unet._block(features * 2, features, name="dec1")

        # Final convolution
        self.conv = nn.Conv2d(
            in_channels=features, out_channels=3, kernel_size=1
        )

    @staticmethod
    def _block(in_channels, out_channels, name):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)

        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)

    def get_loss(self, x_0, t):
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = self(x_t)
        return F.l1_loss(predicted_noise, noise)

    def q_sample(self, x_0, t, noise):
        alpha_bar_t = self.alpha_bar[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise

    @torch.no_grad()
    def p_sample(self, x, t):
        t_index = t
        beta_t = self.beta[t_index]
        alpha_t = self.alpha[t_index]
        alpha_bar_t = self.alpha_bar[t_index]
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t)

        predicted_noise = self(x)

        x_prev = sqrt_recip_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_bar_t) * predicted_noise)
        if t_index > 0:
            noise = torch.randn_like(x)
            sigma_t = torch.sqrt(beta_t)
            x_prev = x_prev + sigma_t * noise

        return x_prev

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = next(self.parameters()).device
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.n_steps)):
            x = self.p_sample(x, t)
        return x


# Initialize model
diffusion = Unet()

# Print the number of trainable parameters
print(f"Total number of trainable parameters: {count_parameters(diffusion)}")

train_loader, test_loader = get_loaders()
# Optimizer
optimizer = optim.Adam(diffusion.parameters(), lr=1e-4)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        optimizer.zero_grad()
        batch_size = images.size(0)
        t = torch.randint(0, diffusion.n_steps, (batch_size,))
        loss = diffusion.get_loss(images, t)
        loss.backward()
        optimizer.step()

    # After training, generate images
    diffusion.eval()
    image_shape = (64, 3, 32, 32)  # Adjust according to your data
    with torch.no_grad():
        generated_images = diffusion.p_sample_loop(image_shape)
        # Rescale to [0,1] if necessary
        generated_images = (generated_images + 1) / 2
        generated_images = torch.clamp(generated_images, 0.0, 1.0)

        # Save the generated images as a grid
    save_image(generated_images, f'fake_images/diffusion/generated_images_epoch_{epoch}.png', nrow=8)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")