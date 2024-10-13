import torch
import torch.nn as nn

import torch.optim as optim
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from architectures.EMA import EMA
from architectures.U_net_embeddings import Unet_with_emb
from get_cifar import get_loaders
from utils import count_parameters


# Initialize model
diffusion = Unet_with_emb().cuda()
ema = EMA(diffusion, decay=0.999)

# Print the number of trainable parameters
print(f"Total number of trainable parameters: {count_parameters(diffusion)}")

train_loader, test_loader = get_loaders()
# Optimizer
optimizer = optim.AdamW(diffusion.parameters(), lr=2e-4, weight_decay=1e-5)
num_epochs = 300
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training loop

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        t = torch.randint(0, diffusion.n_steps, (images.size(0),)).long().cuda()
        optimizer.zero_grad()
        loss = diffusion.get_loss(images, t)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)

        optimizer.step()

        ema.update()

    scheduler.step()
    # After training, generate images
    diffusion.eval()
    image_shape = (64, 3, 32, 32)  # Adjust according to your data
    with torch.no_grad():
        generated_images = diffusion.p_sample_loop(image_shape)
        # Rescale to [0,1] if necessary
        generated_images = (generated_images + 1) / 2
        generated_images = torch.clamp(generated_images, 0.0, 1.0)

    diffusion.train()
        # Save the generated images as a grid
    save_image(generated_images, f'../fake_images/diffusion/generated_images_epoch_{epoch}.png', nrow=8)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")