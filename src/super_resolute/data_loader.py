import os

import torch
from torch.utils.data import DataLoader

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ImageDataset(Dataset):
    """Input High/Low resolution dataset."""

    def __init__(self, root_dir, scale_factor, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = [os.path.join(root_dir, i) for i in os.listdir(root_dir) if i.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_HR = Image.open(self.image_list[idx])
        width, height = image_HR.size
        image_LR = image_HR.resize(
            (width // self.scale_factor, height // self.scale_factor))

        if self.transform:
            image_HR = self.transform(image_HR)
            image_LR = self.transform(image_LR)

        return image_HR, image_LR

def get_super_resolute_data_loader(opts):
    """Creates training, validation data loaders.
    """
    transform = transforms.Compose([
        transforms.RandomCrop(opts.image_size),
        transforms.ToTensor(),
    ])

    train_dataset = ImageDataset(
        root_dir=opts.training_data_path, scale_factor=opts.scale_factor, transform=transform)

    test_dataset = ImageDataset(
        root_dir=opts.validation_data_path, scale_factor=opts.scale_factor, transform=transform)

    train_dloader = DataLoader(
        dataset=train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    valid_dloader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size,
                              shuffle=False, num_workers=opts.num_workers)

    return train_dloader, valid_dloader
