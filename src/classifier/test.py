from model import FruitClassifier
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.autograd import Variable
import argparse
import cv2
import numpy as np
from PIL import Image

transforms_train = T.Compose([T.ToTensor(),T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
image_data_train = ImageFolder("./Training_10",transform=transforms_train)
classes_idx = image_data_train.class_to_idx
classes_named = dict([(value, key) for key, value in classes_idx.items()])

parser = argparse.ArgumentParser(description='Fruit classifying tool')
parser.add_argument('--input',
                    help='Image source')

args = parser.parse_args()

image_name = args.input
image = cv2.imread(image_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = FruitClassifier()
net.load_state_dict(torch.load('./weights/classifierweight.pth'))
net = net.cuda()
net = net.to(device)

image_path = image_name
image1 = Image.open(image_path)
# plt.imshow(image1)
# plt.show()
image1 = np.array(image1)
image1 = cv2.resize(image1, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
image1 = transforms_train(image1)
image1 = Variable(image1, requires_grad=True)
image1 = image1.unsqueeze(0)
predict_single = net(image1.cuda())
predicted = torch.max(predict_single.data, 1)

print("The given image path is", image_path, "The fruit in the image is: ", classes_named[predicted.indices[0].item()])

