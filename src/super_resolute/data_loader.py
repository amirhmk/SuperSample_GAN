import os

import torch
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms


def get_super_resolute_data_loader(emoji_type, opts, train_path, valid_path):
    """Creates training, validation data loaders.
    """
    transform = transforms.Compose([
        transforms.RandomCrop(opts.image_size),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(valid_path, transform)

    train_dloader = DataLoader(
        dataset=train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    valid_dloader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size,
                              shuffle=False, num_workers=opts.num_workers)

    return train_dloader, valid_dloader
