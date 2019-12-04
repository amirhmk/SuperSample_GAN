import random
import pandas as pd
import numpy as np
import torchvision
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from PIL import Image


# Loading data
transforms_train = T.Compose([T.ToTensor(),T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
image_data_train = ImageFolder("./ten/Training_10",transform=transforms_train)
image_data_test = ImageFolder("./ten/Test_10",transform=transforms_train)

# Shuffling data and then collecting all the labels.
random.shuffle(image_data_train.samples)
random.shuffle(image_data_test.samples)

train_loader = DataLoader(dataset=image_data_train,batch_size=300)
test_loader = DataLoader(dataset=image_data_test,batch_size=300)

# Total classes
classes_idx = image_data_train.class_to_idx
classes_named = dict([(value, key) for key, value in classes_idx.items()])
classes = len(image_data_train.classes)
len_train_data = len(image_data_train)
len_test_data = len(image_data_test)


def get_labels():
    labels_train = []  # All the labels
    labels_test = []
    for i in image_data_train.imgs:
        labels_train.append(i[1])

    for j in image_data_test.imgs:
        labels_test.append(j[1])

    return (labels_train, labels_test)


labels_train, labels_test = get_labels()

print(len(train_loader.dataset))
print(len(test_loader.dataset))
print("Total Number of fruit classes: ", classes)
print(classes_idx)
print(classes_named)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

net = Net()
net = net.load_state_dict(torch.load('classifierweight.pth'))
# Print net
print(net)

images = []
for image_path1 in images:
    image_path = image_path1
    image1 = Image.open(image_path)
    plt.imshow(image1)
    plt.show()
    image1 = np.array(image1)
    image1 = cv2.resize(image1, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
    image1 = transforms_train(image1)
    image1 = Variable(image1, requires_grad=True)
    image1 = image1.unsqueeze(0)
    print(image1)
    predict_single = net(image1.cuda())
    predicted = torch.max(predict_single.data, 1)

    pred_val = predict_single.data[0][predicted.indices[0].item()].item()
    values, indices = predict_single.topk(5)
    # print(values, indices)
    print(image_path, classes_named[predicted.indices[0].item()])
    for index in indices[0]:
        print(classes_named[index.item()])
    print("====================")
