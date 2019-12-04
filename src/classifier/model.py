import torch.nn as nn
import torch.nn.functional as F


class FruitClassifier(nn.Module):
    def __init__(self):
        super(FruitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 22 * 22, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # print("input shape", x.shape)
        out = F.relu(self.conv1(x))
        # print("before pool 1", out.size())
        out = self.pool(out)
        # x = self.pool(F.relu(self.conv1(x)))
        # print("after pool 1", out.size())
        out = F.relu(self.conv2(out))
        # print("before pool 2", out.size())
        out = self.pool(out)
        # print("before view", out.size())
        x = out.view(-1, 16 * 22 * 22)
        # x = out.reshape((out.size(0), 16 * 5 * 5))
        # print("after view",x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # print("x", x.shape)
        x = self.fc3(x)
        return x