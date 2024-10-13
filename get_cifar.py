import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def collate_fn(batch):
    data, target = zip(*batch)
    data = torch.stack(data).cuda()
    target = torch.tensor(target).cuda()
    return data, target


def get_loaders():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define transforms
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-100 dataset
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader