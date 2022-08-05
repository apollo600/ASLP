from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # max pooling using a 2*2 window
        x = F.max_pool2d(F.softmax(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.softmax(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all the dimension except the batch dimension
        x = F.softmax(self.fc1(x))
        x = F.softmax(self.fc2(x))
        x = self.fc3(x)
        return x
