
import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, conv_dim):
        super(Discriminator, self).__init__()
        self.conv_dim = conv_dim
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, conv_dim, kernel_size=3, stride=1, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                conv_dim, conv_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_dim),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                conv_dim, conv_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_dim * 2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(
                conv_dim * 2, conv_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_dim * 2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(
                conv_dim * 2, conv_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_dim * 4),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(
                conv_dim * 4, conv_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_dim * 4),
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(
                conv_dim * 4, conv_dim * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_dim * 8),
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(
                conv_dim * 8, conv_dim * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_dim * 8),
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(
                conv_dim * 8, conv_dim * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_dim * 8),
        )
        self.layer10 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                conv_dim * 8, conv_dim * 16, kernel_size=1, stride=1, padding=1),
        )

        self.last = nn.Conv2d(conv_dim * 16, 1, kernel_size=1, stride=1)

    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))        
        out = F.relu(self.layer4(out))
        out = F.relu(self.layer5(out))
        out = F.relu(self.layer6(out))
        out = F.relu(self.layer7(out))
        out = F.relu(self.layer8(out))
        out = F.relu(self.layer9(out))
        out = F.relu(self.layer10(out))
        out = torch.sigmoid(self.last(out).view(self.conv_dim))