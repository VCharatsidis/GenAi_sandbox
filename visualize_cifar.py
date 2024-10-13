import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Define transformations with resizing and normalization
transform = transforms.Compose([
    transforms.Resize(128),  # Resize images to 128x128
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# Load the test dataset
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform)

# Create a DataLoader
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=True)

# Get class names
classes = testset.classes

# Function to unnormalize and show images
def imshow(img):
    img = img * 0.5 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(8, 8))  # Increase figure size
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.axis('off')  # Hide axes
    plt.show()

# Get random test images
dataiter = iter(testloader)
images, labels = next(dataiter)

# Show images
imshow(torchvision.utils.make_grid(images, nrow=8))

# Print labels
print('Labels:')
for idx in range(len(labels)):
    print(f'{idx+1}: {classes[labels[idx]]}')
