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

class ModelC4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4, 16, 3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(4)
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

class ModelC8(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(8)
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

class ModelD(nn.Module):
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

        self.conv7 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn41 = nn.BatchNorm2d(128)
        self.bn42 = nn.BatchNorm2d(128)

        self.dropout1=nn.Dropout(0.3)
        
        #self.fc1 = nn.Linear(32768, 1)
        self.fc1 = nn.Linear(8192, 1)
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

        x = F.relu(self.bn41(self.conv7(x)))
        x = F.relu(self.bn42(self.conv8(x)))
        x = self.pool(x)
        
        x = self.dropout1(x)
        
        x = x.flatten(start_dim=1, end_dim=-1) # flatten output for linear layer

        x = self.fc1(x)
        #x = self.fc2(x)
        return x

class ModelE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(64)
        self.bn12 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn21 = nn.BatchNorm2d(128)
        self.bn22 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn31 = nn.BatchNorm2d(256)
        self.bn32 = nn.BatchNorm2d(256)

        self.dropout1=nn.Dropout(0.3)
        
        #self.fc1 = nn.Linear(32768, 1)
        self.fc1 = nn.Linear(65536, 1)
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


class ModelA2(nn.Module):
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

        self.dropout1=nn.Dropout(0.2)
        
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


class ModelA4(nn.Module):
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

        self.dropout1=nn.Dropout(0.4)
        
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

class PyrA(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, 3, stride=1, padding=1)
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
        self.fc1 = nn.Linear(1024, 1)
        self.fc2 = nn.Linear(256, 1)

        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        #x = x.unsqueeze_(0)
        #x = x.permute(1,0,2,3)

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

class PyrB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 3, stride=1, padding=1)
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
        self.fc1 = nn.Linear(2048, 1)
        self.fc2 = nn.Linear(256, 1)

        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        #x = x.unsqueeze_(0)
        #x = x.permute(1,0,2,3)

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