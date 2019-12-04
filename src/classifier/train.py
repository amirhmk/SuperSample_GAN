from model import FruitClassifier
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# Loading data
transforms_train = T.Compose([T.ToTensor(),T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
image_data_train = ImageFolder("./Training_10",transform=transforms_train)
image_data_test = ImageFolder("./Test_10",transform=transforms_train)

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

print("Total Number of fruit classes: ", classes)
print(classes_idx)
print(classes_named)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

net = FruitClassifier().float()
net = net.cuda()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        batch_size = inputs.size(0)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if i % 20 == 0:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')


total = 0
correct = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on test images:' + str(100 * correct / total))

# Save model weights
torch.save(net.state_dict(), './weights/classifierweight.pth')