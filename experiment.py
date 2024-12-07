#!/usr/bin/env python

# imports and cuda setup

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

savepath="data"
dir = os.path.join(os.path.dirname(__file__), savepath)

f = open(os.path.join(dir, 'experiment_open2.log'),'w')

# unpack wfs data
#turbs = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
#turbs = [0.05, 0.1]
samples = 10000
seed = 2818

codes = ['r0.05_1202', 'r0.06_8449', 'r0.07_5284', 'r0.08_3219', 'r0.09_2483', 'r0.1_9502', 'r0.11_6639', 'r0.12_1687', 'r0.13_8069', 'r0.14_1138', 'r0.15_2556', 'r0.16_1030', 'r0.17_815', 'r0.18_5519', 'r0.19_4963', 'r0.2_6184']
#codes = ['r0.05_8843', 'r0.06_4722', 'r0.07_9303', 'r0.08_90', 'r0.09_8354', 'r0.1_465', 'r0.11_5733', 'r0.12_2390', 'r0.13_2516', 'r0.14_8165', 'r0.15_4511', 'r0.16_753', 'r0.17_2428', 'r0.18_8208', 'r0.19_9645', 'r0.2_9616']

img_data = []
label_data = []

for c in codes:
    path = os.path.join(dir, f'wfs_data_closed_{c}_{samples}.npz')
    #path = os.path.join(dir, f'wfs_data_open_{c}_{samples}.npz')
    container = np.load(path)
    img_data = [*img_data, *container['arr_0']]
    label_data = [*label_data, *container['arr_1']]


# Data Engineering

f_label_data = [str(np.round(ld, 2)) for ld in label_data]
classes = set(np.unique(f_label_data))
#print('classes: ', classes)
#print('# of classes: ', len(classes))

print("data loaded")


# Normlisation

X = img_data / (np.max(img_data)*1.1)


# convert labels to one-hot

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(f_label_data)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

#print('label: ',f_label_data[idx])
#print('one-hot: ',onehot_encoded[idx])

y = onehot_encoded


# Training, Validation, and Test Sets

train_ratio = 0.75
val_ratio = 0.15
test_ratio = 0.10

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_ratio, shuffle=True)

X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + val_ratio), shuffle=True) 

print("train-test split done")

# Make Custom Dataset Class

class ImageDataset(Dataset):
    def __init__(self, lbl, img, transform=None, target_transform=None):
        self.labels = lbl
        self.imgs = img
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
tr_transform = transforms.Compose([
    transforms.ToTensor()
    
    # flip with 50% chance 
    #transforms.RandomHorizontalFlip(0.5), 
])

vl_transform = transforms.Compose([
    transforms.ToTensor()
])


# Parameters
batch_size = 32
epochs = 300
lr = 0.001


# Define a training, validation and test sets
training_set = ImageDataset(
    lbl = y_train,
    img = X_train,
    transform = tr_transform,
    target_transform = vl_transform
)

validation_set = ImageDataset(
    lbl = y_val,
    img = X_val,
    transform = tr_transform,
    target_transform = vl_transform
)

test_set = ImageDataset(
    lbl = y_test,
    img = X_test,
    transform = tr_transform,
    target_transform = vl_transform
)

# Define dataloader for each
train_loader = DataLoader(TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train)), batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val)), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test)), batch_size=batch_size, shuffle=True)

print("dataloaders done")

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(8, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.fc11 = nn.Linear(16384, 128)
        self.fc12 = nn.Linear(128, 64)
        self.fc13 = nn.Linear(64, len(classes))
    def forward(self, x):
        #x = F.relu(self.conv1(x))
        #x = self.pool(x)
        #x = F.relu(self.conv2(x))
        #x = self.pool(x)
        x = x.flatten(start_dim=1, end_dim=-1) # flatten output for linear layer
        #x = F.relu(self.fc1(x))
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc12(x))
        return self.fc13(x)



class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.fc11 = nn.Linear(1024, len(classes))
    def forward(self, x):
        x = x.unsqueeze_(0)
        x = x.permute(1,0,2,3)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(start_dim=1, end_dim=-1) # flatten output for linear layer
        return self.fc11(x)
    


class Model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.fc11 = nn.Linear(512, len(classes))
    def forward(self, x):
        x = x.unsqueeze_(0)
        x = x.permute(1,0,2,3)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.flatten(start_dim=1, end_dim=-1) # flatten output for linear layer
        return self.fc11(x)
    


class Model4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.fc11 = nn.Linear(512, len(classes))
    def forward(self, x):
        x = x.unsqueeze_(0)
        x = x.permute(1,0,2,3)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.flatten(start_dim=1, end_dim=-1) # flatten output for linear layer
        return self.fc11(x)
    

class Model5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.fc11 = nn.Linear(128, len(classes))
    def forward(self, x):
        x = x.unsqueeze_(0)
        x = x.permute(1,0,2,3)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = x.flatten(start_dim=1, end_dim=-1) # flatten output for linear layer
        return self.fc11(x)

class Model6(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.fc11 = nn.Linear(2048, len(classes))
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
        x = x.flatten(start_dim=1, end_dim=-1) # flatten output for linear layer
        return self.fc11(x)

class Model7(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.fc11 = nn.Linear(512, len(classes))
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
        x = F.relu(self.conv7(x))
        x = self.pool(x)
        x = x.flatten(start_dim=1, end_dim=-1) # flatten output for linear layer
        return self.fc11(x)

class Model8(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.fc11 = nn.Linear(512, len(classes))
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
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool(x)
        x = x.flatten(start_dim=1, end_dim=-1) # flatten output for linear layer
        return self.fc11(x)

#net = Model() #Model2()
#net.to(device)

#nets = [Model(), Model2(), Model3(), Model4(), Model5(), Model6(), Model7(), Model8()]
nets = [Model7(), Model8()]
print('loaded models')

# Tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

save_dir = os.path.join(os.path.dirname(__file__), "checkpoints")

# Training loop
print('started training')

for net in nets:
    #net = Model8()
    net.to(device)
    print(net, file=f)
    # Define cross-entropy loss function
    loss_func = nn.CrossEntropyLoss()

    # Define Adam optimizer, with 1e-3 learning rate and betas=(0.9, 0.999).
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    val_scores=[]
    #print(batch_size)
    # loop over the dataset multiple times

    best_val_loss = float('inf')
    best_acc = float('inf')
    best_epoch = 0
    counter = 1
    save_f = os.path.join(save_dir, 'best_model' + str(counter) + '.pth')

    
    for epoch in range(epochs):
        print('Starting training for epoch {}'.format(epoch+1), file=f)
        training_loss = 0.0
        correct, total = 0, 0

        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            #print('in: ', inputs)
            #print("lbl: ", labels)
            #print(inputs.shape)
            #print(labels.shape)

            inputs, labels = inputs.to(device), labels.to(device)
            
            # zero parameter gradients
            optimizer.zero_grad()
            
            # get output based off input
            inputs = inputs.float()
            outputs = net(inputs)
            #print('labels: ', labels)
            #print('output: ', outputs)
            
            # calculate loss and gradients
            loss = loss_func(outputs, labels)
            loss.backward()
            
            # optimise based off gradients
            optimizer.step()
            
            # sum for accuracy
            _, predicted = torch.max(outputs.data, 1)
            #print('predicted: ', predicted)
            #print('labels: ', torch.argmax(labels, axis=1))
            total += labels.size(0)
            batch_correct = (predicted == torch.argmax(labels, axis=1)).sum().item()
            correct += batch_correct
            #print('# correct in batch: ', batch_correct)
            training_loss += loss.item()
        
        # display and record training loss + accuracy
        training_loss /= len(train_loader)
        print('Training loss for epoch {:2d}: {:5f}'.format(epoch+1, training_loss), file=f)
        writer.add_scalar("Loss/train", training_loss, epoch)
        
        print(f'Accuracy of the network on the training images: {100 * correct // total} %', file=f)
        writer.add_scalar("Accuracy/train", 100 * correct / total, epoch)
        
        # save model at end of each epoch
        #PATH = os.path.join(dir, 'checkpoints/net_{:02d}.pth'.format(epoch))
        #torch.save(net.state_dict(), PATH)
        
        # validation
        validation_loss = 0.0
        v_correct, v_total = 0, 0
        print('Starting validation for epoch {}'.format(epoch+1), file=f)
        with torch.no_grad():
            for data in validation_loader:
                # get inputs
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                # get output based off input
                inputs = inputs.float()
                outputs = net(inputs)
                
                # calculate loss
                loss = loss_func(outputs, labels)
                
                # summate loss and accuracy data
                validation_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                v_total += labels.size(0)
                v_correct += (predicted == torch.argmax(labels, axis=1)).sum().item()
                
            # display and record validation loss + accuracy
            validation_loss /= len(validation_loader)
            print('Validation loss for epoch {:2d}: {:5f}'.format(epoch+1, validation_loss), file=f)
            writer.add_scalar("Loss/val", validation_loss, epoch)
            
            print(f'Accuracy of the network on the validation images: {100 * v_correct // v_total} %', file=f)
            writer.add_scalar("Accuracy/val", 100 * v_correct / v_total, epoch)

            # Save the best model based on validation loss
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                best_acc = 100 * v_correct / v_total
                best_epoch = epoch
                torch.save(net.state_dict(), save_f)
        print(f"Epoch [{best_epoch+1}/{epochs}] - "f"{training_loss:.4f}/{validation_loss:.4f}/{100 * correct / total:.4f}/{100 * v_correct / v_total:.4f}", file=f)
        
        print(f"Best Result: Epoch [{best_epoch+1}/{epochs}] - "f"Val Loss: {best_val_loss:.4f}, Val Error: {best_acc:.4f}", file=f)
    
    counter += 1
    print('Finished Training', file=f)

# Testing
#best_path = './checkpoints/net_29.pth'

#net = Model()
#net.load_state_dict(torch.load(best_path))
#net.to(device)

    # Test the model on test set
    correct, total = 0, 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.argmax(labels, axis=1)).sum().item()
            #correct += (predicted == labels).sum().item()
            
    print(f'Accuracy of the network on the test images: {100 * correct / total} %', file=f)