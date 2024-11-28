# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torch.utils.data import sampler
# import torchvision.datasets as dset
# import torchvision.transforms as T
# import numpy as np

# # set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('Using device:', device)

# NUM_TRAIN = 49000
# # The torchvision.transforms package provides tools for preprocessing data
# # and for performing data augmentation; here we set up a transform to
# # preprocess the data by subtracting the mean RGB value and dividing by the
# # standard deviation of each RGB value; we've hardcoded the mean and std.
# transform = T.Compose([
#     T.ToTensor(),
#     T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#     ])

# # create galaxy map later?

import torch
import torch.nn as nn
import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            # Adjust based on image size after convolutions
            nn.Linear(64 * 32 * 32, 128),  
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# net = Net()
    

    # def __init__(self, feature_dim=128):
    #     super(Model, self).__init__()

    #     self.f = []
    #     for name, module in resnet50().named_children():
    #         if name == 'conv1':
    #             module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    #         if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
    #             self.f.append(module)
    #     # encoder
    #     self.f = nn.Sequential(*self.f)
    #     # projection head
    #     self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
    #                            nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    # def forward(self, x):
    #     x = self.f(x)
    #     feature = torch.flatten(x, start_dim=1)
    #     out = self.g(feature)
    #     return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)