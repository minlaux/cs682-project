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