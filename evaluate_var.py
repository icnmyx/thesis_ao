import numpy as np
import closed_loop_r0 as clr
from docopt import docopt
import os

import matplotlib.pyplot as plt
import random
#from experiment_int_test import Model, ImageDataset
import torch
from collections import deque
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
import torch.nn as nn
from models import Model, Model2
from datetime import datetime

"""
script to get avg and var of model against whole dataset

"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
isCategorical = False

savepath="data"
dir = os.path.join(os.path.dirname(__file__), savepath)

# Load Data
samples = 10000
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

f_label_data = [np.round(ld, 2) for ld in label_data]
classes = set(np.unique(f_label_data))
print("data loaded")

# Normlisation

X = img_data / (np.max(img_data)*1.1)

if isCategorical:
  # convert labels to one-hot
  label_encoder = LabelEncoder()
  integer_encoded = label_encoder.fit_transform(f_label_data)
  integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

  onehot_encoder = OneHotEncoder(sparse=False)
  onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
  y = onehot_encoded

else: 
  y = f_label_data

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
        
transform = transforms.Compose([transforms.ToTensor()])
training_set = ImageDataset(
    lbl = y,
    img = X,
    transform = transform,
    target_transform = transform
)
batch_size = 256
dl = DataLoader(TensorDataset(torch.Tensor(X), torch.Tensor(y)), batch_size=batch_size, shuffle=True)

# Load Model
best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), 'best_model_0206_1.pth')
net = Model2()
net.load_state_dict(torch.load(best_path))
net.to(device)

# Eval
if __name__ == "__main__":
  # Function to evaluate the model on the entire dataset
  def evaluate_model(model, data_loader):
      #criterion = nn.MSELoss()
      criterion = nn.L1Loss()
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      model = model.to(device)
      model.eval()

      total_loss = 0.0
      predictions = []
      predictions_by_label = {}
      with torch.no_grad():
          for i, data in enumerate(dl, 0):
              inputs, labels = data
              inputs, labels = inputs.to(device), labels.to(device)

              inputs = inputs.float()
              outputs = torch.flatten(model(inputs))

              loss = criterion(outputs, labels)
              total_loss += loss.item()

              if isCategorical:
                 _, predicted = torch.max(outputs.data, 1)
                 predicted = predicted.cpu().numpy()
              else:
                 predicted = outputs.cpu().numpy()
              
              
              for i in range(len(labels)):
                  label = labels[i].item()
                  if label not in predictions_by_label:
                      predictions_by_label[label] = []
                  predictions_by_label[label].append(predicted[i])

              predictions.extend(predicted)

      avg_loss = total_loss / len(data_loader)
      return avg_loss, predictions, predictions_by_label
  

  # Evaluate the model on the entire dataset
  avg_loss, predictions, predictions_by_label = evaluate_model(net, dl)

  # Measure variance of the predictions
  #variance = torch.var(torch.tensor(predictions)).item()

  # Print average loss and variance
  print("Average Loss:", avg_loss)
  #print("Variance of Predictions:", variance)
  
  #print("Variance of Predictions by Label:")

  # Visualize the distribution of predictions for each label
  plt.figure(figsize=(10, 6))
  for label, pred in sorted(predictions_by_label.items(), key=lambda item: item[1]):
    plt.hist(pred, bins=len(classes), alpha=0.5, label=f"{np.round(label,2)}")
    plt.axvline(label, alpha=0.5, color='k', linestyle='dashed', linewidth=1)
    #print(f"r0: {np.round(label,2)} - Variance: {torch.var(torch.tensor(pred)).item()}")

  plt.xlabel("Predicted Value")
  plt.ylabel("Frequency")
  plt.title("Distribution of Predictions by True r0")
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  # Get the current timestamp
  current_time = datetime.now().strftime("%Y%m%d%H%M%S")

  # Save the plot as an image with the current time as the file name
  file_name = f"eval_var_{current_time}.png"
  plt.savefig(file_name)

  plt.show()




