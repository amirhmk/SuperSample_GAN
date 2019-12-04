from model import FruitClassifier
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.autograd import Variable
import argparse
# import cv2
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
image = Image.open(image_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = FruitClassifier()
image_path = image_name
image1 = Image.open(image_path)
image1 = np.array(image1)
image1 = image.resize((100, 100), Image.BICUBIC)
image1 = transforms_train(image1)
image1 = Variable(image1, requires_grad=True)
image1 = image1.unsqueeze(0)
print(device)
if str(device) == 'cpu':
    print('got t')
    net = net.to(device)
    net.load_state_dict(torch.load(
        './weights/classifierweight.pth', map_location=torch.device('cpu')))
else:
    net = net.cuda()
    net.load_state_dict(torch.load('./weights/classifierweight.pth'))
    image1 = image.cuda()


# plt.imshow(image1)
# plt.show()
predict_single = net(image1)
predicted = torch.max(predict_single.data, 1)

print("The given image path is", image_path, "The fruit in the image is: ", classes_named[predicted.indices[0].item()])

