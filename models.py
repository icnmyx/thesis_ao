import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch

class ModelA0(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(16)
        self.bn12 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn21 = nn.BatchNorm2d(32)
        self.bn22 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn31 = nn.BatchNorm2d(64)
        self.bn32 = nn.BatchNorm2d(64)

        self.dropout1=nn.Dropout(0.1)
        
        #self.fc1 = nn.Linear(32768, 1)
        self.fc1 = nn.Linear(16384, 1)
        self.fc2 = nn.Linear(256, 1)

        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = x.unsqueeze_(0)
        x = x.permute(1,0,2,3)

        x = F.relu(self.bn11(self.conv1(x)))
        x = F.relu(self.bn12(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn21(self.conv3(x)))
        x = F.relu(self.bn22(self.conv4(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn31(self.conv5(x)))
        x = F.relu(self.bn32(self.conv6(x)))
        x = self.pool(x)
        
        #x = self.dropout1(x)
        
        x = x.flatten(start_dim=1, end_dim=-1) # flatten output for linear layer

        x = self.fc1(x)
        #x = self.fc2(x)
        return x

class ModelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(16)
        self.bn12 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn21 = nn.BatchNorm2d(32)
        self.bn22 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn31 = nn.BatchNorm2d(64)
        self.bn32 = nn.BatchNorm2d(64)

        self.dropout1=nn.Dropout(0.1)
        
        #self.fc1 = nn.Linear(32768, 1)
        self.fc1 = nn.Linear(16384, 1)
        self.fc2 = nn.Linear(256, 1)

        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = x.unsqueeze_(0)
        x = x.permute(1,0,2,3)

        x = F.relu(self.bn11(self.conv1(x)))
        x = F.relu(self.bn12(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn21(self.conv3(x)))
        x = F.relu(self.bn22(self.conv4(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn31(self.conv5(x)))
        x = F.relu(self.bn32(self.conv6(x)))
        x = self.pool(x)
        
        x = self.dropout1(x)
        
        x = x.flatten(start_dim=1, end_dim=-1) # flatten output for linear layer

        x = self.fc1(x)
        #x = self.fc2(x)
        return x


class ModelA3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(16)
        self.bn12 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn21 = nn.BatchNorm2d(32)
        self.bn22 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn31 = nn.BatchNorm2d(64)
        self.bn32 = nn.BatchNorm2d(64)

        self.dropout1=nn.Dropout(0.3)
        
        #self.fc1 = nn.Linear(32768, 1)
        self.fc1 = nn.Linear(16384, 1)
        self.fc2 = nn.Linear(256, 1)

        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = x.unsqueeze_(0)
        x = x.permute(1,0,2,3)

        x = F.relu(self.bn11(self.conv1(x)))
        x = F.relu(self.bn12(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn21(self.conv3(x)))
        x = F.relu(self.bn22(self.conv4(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn31(self.conv5(x)))
        x = F.relu(self.bn32(self.conv6(x)))
        x = self.pool(x)
        
        x = self.dropout1(x)
        
        x = x.flatten(start_dim=1, end_dim=-1) # flatten output for linear layer

        x = self.fc1(x)
        #x = self.fc2(x)
        return x

class ModelC(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(32)
        self.bn12 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn21 = nn.BatchNorm2d(64)
        self.bn22 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn31 = nn.BatchNorm2d(128)
        self.bn32 = nn.BatchNorm2d(128)

        self.dropout1=nn.Dropout(0.3)
        
        #self.fc1 = nn.Linear(32768, 1)
        self.fc1 = nn.Linear(32768, 1)
        self.fc2 = nn.Linear(256, 1)

        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = x.unsqueeze_(0)
        x = x.permute(1,0,2,3)

        x = F.relu(self.bn11(self.conv1(x)))
        x = F.relu(self.bn12(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn21(self.conv3(x)))
        x = F.relu(self.bn22(self.conv4(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn31(self.conv5(x)))
        x = F.relu(self.bn32(self.conv6(x)))
        x = self.pool(x)
        
        x = self.dropout1(x)
        
        x = x.flatten(start_dim=1, end_dim=-1) # flatten output for linear layer

        x = self.fc1(x)
        #x = self.fc2(x)
        return x

class ModelA5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(16)
        self.bn12 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn21 = nn.BatchNorm2d(32)
        self.bn22 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn31 = nn.BatchNorm2d(64)
        self.bn32 = nn.BatchNorm2d(64)

        self.dropout1=nn.Dropout(0.5)
        
        #self.fc1 = nn.Linear(32768, 1)
        self.fc1 = nn.Linear(16384, 1)
        self.fc2 = nn.Linear(256, 1)

        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = x.unsqueeze_(0)
        x = x.permute(1,0,2,3)

        x = F.relu(self.bn11(self.conv1(x)))
        x = F.relu(self.bn12(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn21(self.conv3(x)))
        x = F.relu(self.bn22(self.conv4(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn31(self.conv5(x)))
        x = F.relu(self.bn32(self.conv6(x)))
        x = self.pool(x)
        
        x = self.dropout1(x)
        
        x = x.flatten(start_dim=1, end_dim=-1) # flatten output for linear layer

        x = self.fc1(x)
        #x = self.fc2(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.dropout1=nn.Dropout(0.1)
        
        #self.fc1 = nn.Linear(32768, 1)
        self.fc1 = nn.Linear(16384, 1)
        self.fc2 = nn.Linear(256, 1)

        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = x.unsqueeze_(0)
        x = x.permute(1,0,2,3)

        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn1(F.relu(self.conv2(x)))
        x = self.pool(x)
        
        x = self.bn2(F.relu(self.conv3(x)))
        x = self.bn2(F.relu(self.conv4(x)))
        x = self.pool(x)
        
        x = self.bn3(F.relu(self.conv5(x)))
        x = self.bn3(F.relu(self.conv6(x)))
        x = self.pool(x)
        
        x = self.dropout1(x)
        
        x = x.flatten(start_dim=1, end_dim=-1) # flatten output for linear layer

        x = self.fc1(x)
        #x = self.fc2(x)
        return x

class ModelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.dropout1=nn.Dropout(0.1)
        
        #self.fc1 = nn.Linear(32768, 1)
        self.fc1 = nn.Linear(16384, 1)
        self.fc2 = nn.Linear(256, 1)

        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = x.unsqueeze_(0)
        x = x.permute(1,0,2,3)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        
        x = self.dropout1(x)
        
        x = x.flatten(start_dim=1, end_dim=-1) # flatten output for linear layer

        x = self.fc1(x)
        #x = self.fc2(x)
        return x
        
class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.dropout1=nn.Dropout(0.1)
        
        self.fc1 = nn.Linear(16384, 1)
        self.fc2 = nn.Linear(256, 1)

        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = x.unsqueeze_(0)
        x = x.permute(1,0,2,3)

        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool(x)
        
        x = self.bn2(F.relu(self.conv3(x)))
        x = self.pool(x)
        
        x = self.bn3(F.relu(self.conv5(x)))
        x = self.pool(x)
        
        x = self.dropout1(x)
        
        x = x.flatten(start_dim=1, end_dim=-1) # flatten output for linear layer

        x = self.fc1(x)
        #x = self.fc2(x)
        return x

class Model4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.dropout1=nn.Dropout(0.1)
        
        self.fc1 = nn.Linear(16384, 1)
        self.fc2 = nn.Linear(256, 1)

        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = x.unsqueeze_(0)
        x = x.permute(1,0,2,3)

        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn1(F.relu(self.conv2(x)))
        x = self.pool(x)
        
        x = self.bn2(F.relu(self.conv3(x)))
        x = self.bn2(F.relu(self.conv4(x)))
        x = self.pool(x)
        
        x = self.bn3(F.relu(self.conv5(x)))
        x = self.bn3(F.relu(self.conv6(x)))
        x = self.pool(x)
        
        x = self.dropout1(x)
        
        x = x.flatten(start_dim=1, end_dim=-1) # flatten output for linear layer

        x = self.fc1(x)
        #x = self.fc2(x)
        return x

# Define the basic residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection if the number of channels changes
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Add the shortcut connection
        out += self.shortcut(identity)
        out = self.relu(out)

        return out


# Define the ResNet model
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1)

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x