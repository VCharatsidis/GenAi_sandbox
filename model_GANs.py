import torch.nn as nn
import torch.optim as optim
import torch
from get_cifar import get_loaders
from torchvision.utils import save_image

from utils import count_parameters


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


# Initialize models
latent_dim = 100
generator = Generator(latent_dim)
discriminator = Discriminator()

# Print the number of trainable parameters
print(f"Generator number of trainable parameters: {count_parameters(generator)}")
print(f"Discriminator number of trainable parameters: {count_parameters(discriminator)}")

# Loss function and optimizers
criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

train_loader, test_loader = get_loaders()

# Create fixed noise for consistent image generation
fixed_noise = torch.randn(64, latent_dim, 1, 1)

# Training loop (simplified)
num_epochs = 100
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):

        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size)
        fake_labels = torch.zeros(batch_size)

        noise = torch.randn(batch_size, latent_dim, 1, 1)
        fake_images = generator(noise)

        # Train Discriminator
        d_optimizer.zero_grad()

        outputs = discriminator(real_images)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()

        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()

        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        g_optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss_real.item():.4f}, g_loss: {g_loss.item():.4f}')

    # Save generated images
    with torch.no_grad():
        fake_images = generator(fixed_noise).detach().cpu()
        # Denormalize the images from [-1, 1] to [0, 1]
        fake_images = (fake_images + 1) / 2
        save_image(fake_images, f'fake_images/gans/fake_images_epoch_{epoch + 1}.png', nrow=8)
