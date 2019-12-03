import os

import torch
from torch.utils.data import DataLoader

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ImageDataset(Dataset):
    """Input High/Low resolution dataset."""

    def __init__(self, root_dir, image_size, scale_factor, transformHR=None, transformLR=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.transform_HR = transformHR
        self.transform_LR = transformLR
        self.image_list = [os.path.join(root_dir, i) for i in os.listdir(root_dir) if i.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
        self.image_size = image_size
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_HR = Image.open(self.image_list[idx])
        image_LR = image_HR.resize(
            (self.image_size // self.scale_factor, self.image_size // self.scale_factor), Image.BICUBIC)

        if self.transform_HR:
            image_HR = self.transform_HR(image_HR)
            image_LR = self.transform_LR(image_LR)

        return image_HR, image_LR

def get_super_resolute_data_loader(opts):
    """Creates training, validation data loaders.
    """
    transformHR = transforms.Compose([
        transforms.RandomCrop(opts.image_size),
        transforms.ToTensor(),
    ])

    transformLR = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = ImageDataset(
        root_dir=opts.training_data_path, image_size=opts.image_size, scale_factor=opts.scale_factor, transformHR=transformHR, transformLR=transformLR)

    test_dataset = ImageDataset(
        root_dir=opts.validation_data_path, image_size=opts.image_size, scale_factor=opts.scale_factor, transformHR=transformHR, transformLR=transformLR)

    train_dloader = DataLoader(
        dataset=train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    valid_dloader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size,
                              shuffle=False, num_workers=opts.num_workers)

    return train_dloader, valid_dloader
