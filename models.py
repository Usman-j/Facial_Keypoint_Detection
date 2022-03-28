## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16,32,3)
        self.conv3 = nn.Conv2d(32,64,3)
        self.conv4 = nn.Conv2d(64,128,3)
        self.conv5 = nn.Conv2d(128,256,3)
        self.fc1   = nn.Linear(256*5*5,500)
        self.fc2   = nn.Linear(500,500)
        self.fc3   = nn.Linear(500,500)
        self.fc4   = nn.Linear(500,500)
        self.fc5   = nn.Linear(500,68*2)
        self.drop1 = nn.Dropout2d(p=0.1)
        self.drop2 = nn.Dropout2d(p=0.2)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = self.drop1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = self.drop2(x)
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        x = F.max_pool2d(F.relu(self.conv4(x)), (2,2))

        x = F.max_pool2d(F.relu(self.conv5(x)), (2,2))

#         print(x.size())
        x = x.view(x.size(0), -1)
#         print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
    
class Net_1(nn.Module):

    def __init__(self):
        super(Net_1, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32,3)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,64,3)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,128,3)
        self.bn4   = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128,256,3)
        self.bn5   = nn.BatchNorm2d(256)
        self.fc1   = nn.Linear(256*5*5,500)
        self.bn6   = nn.BatchNorm1d(500)
        self.fc2   = nn.Linear(500,68*2)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), (2,2))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (2,2))
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), (2,2))
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), (2,2))
        x = F.max_pool2d(F.relu(self.bn5(self.conv5(x))), (2,2))
#         print(x.size())
        x = x.view(x.size(0), -1)
#         print(x.size())
        x = F.relu(self.bn6(self.fc1(x)))
        x = self.fc2(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x