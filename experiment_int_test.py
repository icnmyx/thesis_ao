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
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dir = os.path.join(os.path.dirname(__file__), "data")
savepath = os.path.join(os.path.dirname(__file__), "results")

f = open(os.path.join(savepath, 'experiment_int_test_7.log'),'w')

# unpack wfs data
#turbs = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
#turbs = [0.05, 0.1]
closed = True
samples = 10000
seed = 1234

if closed:
    codes = ['r0.05_1202', 'r0.06_8449', 'r0.07_5284', 'r0.08_3219', 'r0.09_2483', 'r0.1_9502', 'r0.11_6639', 'r0.12_1687', 'r0.13_8069', 'r0.14_1138', 'r0.15_2556', 'r0.16_1030', 'r0.17_815', 'r0.18_5519', 'r0.19_4963', 'r0.2_6184']
else:
    codes = ['r0.05_8843', 'r0.06_4722', 'r0.07_9303', 'r0.08_90', 'r0.09_8354', 'r0.1_465', 'r0.11_5733', 'r0.12_2390', 'r0.13_2516', 'r0.14_8165', 'r0.15_4511', 'r0.16_753', 'r0.17_2428', 'r0.18_8208', 'r0.19_9645', 'r0.2_9616']

img_data = []
label_data = []

for c in codes:
    if closed: 
        path = os.path.join(dir, f'wfs_data_closed_{c}_{samples}.npz')
    else:
        path = os.path.join(dir, f'wfs_data_open_{c}_{samples}.npz')
    container = np.load(path)
    img_data = [*img_data, *container['arr_0']]
    label_data = [*label_data, *container['arr_1']]


# Data Engineering

f_label_data = [np.round(ld, 2) for ld in label_data]
classes = set(np.unique(f_label_data))
print("data loaded")


# Normlisation

X = img_data / (np.max(img_data)*1.1)
y = f_label_data

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
])

vl_transform = transforms.Compose([
    transforms.ToTensor()
])


# Parameters
batch_size = 64
epochs = 300
lr = 0.002


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

from models import Model, Model2, ModelB, ModelA
    
#net = Model() #Model2()
#net.to(device)

#nets = [Model2(), Model()]
nets = [ModelA(), Model()]#, Model3()]
#lrs = [0.001, 0.002, 0.004, 0.008]
print('loaded models')

save_dir = os.path.join(os.path.dirname(__file__), "checkpoints")

# Training loop
print('started training')

counter = 1
for net in nets:
    net.to(device)
    print(net, file=f)
    # Define cross-entropy loss function
    loss_func = nn.L1Loss()

    # Define Adam optimizer, with 1e-3 learning rate and betas=(0.9, 0.999).
    #optimizer = torch.optim.Adam(net.parameters(), lr=lrs[counter-1], betas=(0.9, 0.999))
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    train_losses = []
    val_losses=[]
    train_errors = []
    val_errors=[]

    # loop over the dataset multiple times

    best_val_loss = float('inf')
    best_avg_error = float('inf')
    best_epoch = 0
    save_f = os.path.join(save_dir, 'best_model' + str(counter) + '.pth')

    print('--', file=f)
    print('Experiment Parameters:', file=f)
    print(f"Loss Func: L1 - Closed/Open: {'Closed' if closed else 'Open'} - Batch Size: {batch_size} - Epochs: {epochs} - Learning Rate: {lr}", file=f)
    print('--', file=f)

    for epoch in range(epochs):
        if epoch % 10 == 0: 
            print('Starting training for epoch {}'.format(epoch+1))

        #print('Starting training for epoch {}'.format(epoch+1), file=f)
        training_loss = 0.0
        correct, total = 0, 0
        train_error = 0.0

        net.train()
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
            outputs = torch.flatten(net(inputs))
            #print('labels: ', labels)
            #print('output: ', outputs)
            
            # calculate loss and gradients
            loss = loss_func(outputs, labels)
            loss.backward()
            
            # optimise based off gradients
            optimizer.step()
            
            # sum for accuracy
            diff = torch.abs(outputs - labels)
            #print(diff)
            train_error += torch.sum(diff)

            training_loss += loss.item()
        
        # display and record training loss + accuracy
        training_loss /= len(train_loader)
        train_error /= len(train_loader.dataset)
        
        # validation
        validation_loss = 0.0
        correct, total = 0, 0
        val_error = 0.0

        net.eval()
        #print('Starting validation for epoch {}'.format(epoch+1), file=f)
        with torch.no_grad():
            for data in validation_loader:
                # get inputs
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                # get output based off input
                inputs = inputs.float()
                outputs = torch.flatten(net(inputs))
                
                # calculate loss
                loss = loss_func(outputs, labels)
                
                # summate loss and accuracy data
                validation_loss += loss.item()
                val_error += torch.sum(torch.abs(outputs - labels))
                
            # display and record validation loss + accuracy
            validation_loss /= len(validation_loader)
            val_error /= len(validation_loader.dataset)
            #print('Validation loss for epoch {:2d}: {:5f}'.format(epoch+1, validation_loss), file=f)
            #riter.add_scalar("Loss/val", validation_loss, epoch)
            
            #print(f'Average error with expected validation values: {val_error}', file=f)
            #writer.add_scalar("Error/val", val_error, epoch)
        
            # Save the best model based on validation loss
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                best_avg_error = val_error
                best_epoch = epoch
                torch.save(net.state_dict(), save_f)

        #train_error = train_error.detach().numpy()
        train_losses.append(training_loss)
        val_losses.append(validation_loss)
        train_errors.append(train_error.item())
        val_errors.append(val_error.item())

        print(f"Epoch [{epoch+1}/{epochs}] - "f"{training_loss:.4f}/{validation_loss:.4f}/{train_error:.4f}/{val_error:.4f}", file=f)
        
        if (epoch+1) % 10 == 0 and epoch != 0: 
            print('--', file=f)
            print(f"Best Result: Epoch [{best_epoch+1}/{epochs}] - "f"Val Loss: {best_val_loss:.4f}, Val Error: {best_avg_error:.4f}", file=f)
            print('--', file=f)
    counter += 1

    print('Finished Training', file=f)

    # Testing
    #best_path = './checkpoints/net_29.pth'
    net.load_state_dict(torch.load(save_f))
    net.to(device)

    net.eval()

    # Test the model on test set
    correct, total = 0, 0
    error = 0.0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()
            outputs = torch.flatten(net(inputs))
            error += torch.sum(torch.abs(outputs - labels))

            #correct += (predicted == labels).sum().item()
            
    print(f'Average error with expected test values: {error / len(test_loader.dataset)}', file=f)


    def plot_loss(train_loss, val_loss):
        epochs = range(1, len(train_loss) + 1)

        plt.clf()

        plt.plot(epochs, train_loss, 'b', label='Training Loss')
        plt.plot(epochs, val_loss, 'r', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Get the current timestamp
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")

        # Save the plot as an image with the current time as the file name
        file_name = f"loss_plot_{current_time}.png"
        plt.savefig(file_name)
        plt.show()

    def plot_error(train_error, val_error):
        epochs = range(1, len(train_error) + 1)

        plt.clf()

        plt.plot(epochs, train_error, 'b', label='Training Error')
        plt.plot(epochs, val_error, 'r', label='Validation Error')
        plt.title('Training and Validation Error')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Get the current timestamp
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")

        # Save the plot as an image with the current time as the file name
        file_name = f"error_plot_{current_time}.png"
        plt.savefig(file_name)
        plt.show()

    plot_loss(train_losses, val_losses)
    plot_error(train_errors, val_errors)


# Set the path to the best model checkpoint
#best_path = './checkpoints/net_29.pth'

# Load the best model
#net = Model()
#net.load_state_dict(torch.load(best_path))

# Evaluate the model on the entire dataset
#avg_loss, predictions, predictions_by_label = evaluate_model(model, data_loader)

# Measure variance of the predictions
#variance = torch.var(torch.tensor(predictions)).item()

# Print average loss and variance
#print("Average Loss:", avg_loss)
#print("Variance of Predictions:", variance)

# Visualize the distribution of predictions for each label
# plt.figure(figsize=(10, 6))
# for label, predictions in predictions_by_label.items():
#     plt.hist(predictions, bins=20, alpha=0.5, label=f"Label {label}")

# plt.xlabel("Predicted Value")
# plt.ylabel("Frequency")
# plt.title("Distribution of Predictions by Label")
# plt.legend()
# plt.show()


# Function to evaluate the model on the entire dataset
def evaluate_model(model, data_loader):
    criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    predictions = []
    predictions_by_label = {}
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = outputs.cpu().numpy()
            
            for i in range(len(labels)):
                label = labels[i].item()
                if label not in predictions_by_label:
                    predictions_by_label[label] = []
                predictions_by_label[label].append(predicted[i])

            predictions.extend(pred)

    avg_loss = total_loss / len(data_loader)
    return avg_loss, predictions, predictions_by_label

