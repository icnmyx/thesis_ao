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
from models import Model, Model2, ModelA
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
#codes = ['r0.05_1202', 'r0.06_8449', 'r0.07_5284', 'r0.08_3219', 'r0.09_2483', 'r0.1_9502', 'r0.11_6639', 'r0.12_1687', 'r0.13_8069', 'r0.14_1138', 'r0.15_2556', 'r0.16_1030', 'r0.17_815', 'r0.18_5519', 'r0.19_4963', 'r0.2_6184']
codes = ['r0.05_8843', 'r0.06_4722', 'r0.07_9303', 'r0.08_90', 'r0.09_8354', 'r0.1_465', 'r0.11_5733', 'r0.12_2390', 'r0.13_2516', 'r0.14_8165', 'r0.15_4511', 'r0.16_753', 'r0.17_2428', 'r0.18_8208', 'r0.19_9645', 'r0.2_9616']
img_data = []
label_data = []

for c in codes:
    #path = os.path.join(dir, f'wfs_data_closed_{c}_{samples}.npz')
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
'''
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
'''

batch_size = 64
dl = DataLoader(TensorDataset(torch.Tensor(X), torch.Tensor(y)), batch_size=batch_size)

# Load Model
best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), 'best_model_1906_1.pth')
net = ModelA()
net.load_state_dict(torch.load(best_path))
net.to(device)
net.eval()

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
                 _, predicted = torch.max(outputs.data, 1).cpu().numpy()
              else:
                 predicted = outputs.cpu().numpy()

              #print(predicted)
              #print(labels)
              
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

  thresholds = [0.001, 0.002, 0.005, 0.01]
  
  # Visualize the distribution of predictions for each label
  plt.figure(figsize=(10, 6))
  res = {}
  for label, pred in sorted(predictions_by_label.items(), key=lambda item: item[1]):

    values, bins, _ = plt.hist(pred, bins='auto', alpha=0.5, label=f"{np.round(label,2)}")
    plt.axvline(label, alpha=0.5, color='k', linestyle='dashed', linewidth=1)

    total_values = len(pred)

    print(f"r0: {round(label, 2)} - avg: {np.mean(pred)} - var: {np.var(pred)}")

    
    for t in thresholds:
        lower = label - t
        upper = label + t
        in_range = np.sum([1 if (p >= lower) & (p <= upper) else 0 for p in pred])
        perc = (in_range / total_values) * 100
        #print(f"{perc:.2f}% of predictions within range {t}")
        if str(t) in res:
          temp = res[str(t)]
          temp.append(round(perc, 2))
          res[str(t)] = temp
        else:
          res[str(t)] = [round(perc, 2)]

    #for i in range(len(percentage_dist)):
    #    plt.text(bins[i], percentage_dist[i] + 1, f'{percentage_dist[i]:.1f}%', ha='center',va='bottom', fontsize=8)
    
    #print(f"r0: {np.round(label,2)} - Variance: {torch.var(torch.tensor(pred)).item()}")

  print(res)
  plt.xlabel("Predicted Value")
  plt.ylabel("Frequency")
  plt.title("Distribution of Predictions by True r0")
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  # Get the current timestamp
  current_time = datetime.now().strftime("%Y%m%d%H%M")

  # Save the plot as an image with the current time as the file name
  file_name = f"eval_var_{current_time}.png"
  plt.savefig(file_name)

  plt.show()


'''
0206
Average Loss: 0.002045665226550773
r0: 0.05 - avg: 0.05043439194560051 - var: 3.5068023862550035e-06
r0: 0.06 - avg: 0.06001146882772446 - var: 4.130076831643237e-06
r0: 0.07 - avg: 0.0701899304986 - var: 3.7550942124653375e-06
r0: 0.08 - avg: 0.07998499274253845 - var: 3.7704151054640533e-06
r0: 0.09 - avg: 0.09025859087705612 - var: 3.2425195968244225e-06
r0: 0.1 - avg: 0.10032706707715988 - var: 4.0003137655730825e-06
r0: 0.11 - avg: 0.11041288077831268 - var: 4.572123089019442e-06
r0: 0.12 - avg: 0.12044746428728104 - var: 5.35617118657683e-06
r0: 0.13 - avg: 0.13023741543293 - var: 6.2963986238173675e-06
r0: 0.14 - avg: 0.14046548306941986 - var: 7.427729087794432e-06
r0: 0.15 - avg: 0.1504669338464737 - var: 8.831955710775219e-06
r0: 0.16 - avg: 0.16026423871517181 - var: 1.0302168448106386e-05
r0: 0.17 - avg: 0.17022451758384705 - var: 1.149717081716517e-05
r0: 0.18 - avg: 0.1802651733160019 - var: 1.1531071322679054e-05
r0: 0.19 - avg: 0.1897476315498352 - var: 1.0973627468047198e-05
r0: 0.2 - avg: 0.1986747533082962 - var: 9.775171747605782e-06
{'0.001': [39.73, 38.35, 39.90, 40.30, 42.6, 39.68, 35.78, 33.43, 31.94, 28.44, 26.55, 25.37, 22.89, 23.44, 24.15, 24.15],
 '0.002': [70.17, 67.9, 70.35, 71.27, 73.13, 68.36, 64.34, 61.0, 57.88, 53.22, 50.1, 48.24, 43.96, 44.52, 46.26, 45.64],
 '0.005': [99.19, 98.42, 98.75, 98.67, 99.4, 98.56, 97.7, 96.7, 95.3, 92.76, 90.19, 87.86, 86.04, 85.54, 87.46, 86.07],
 '0.01': [100.0, 100.0, 100.0, 99.99, 100.0, 100.0, 100.0, 100.0, 100.0, 99.97, 99.86, 99.72, 99.71, 99.73, 99.58, 99.6]}

1506 (closed)
Average Loss: 0.0016999023599317297
r0: 0.05 - avg: 0.04961829259991646 - var: 1.4405572983378079e-06
r0: 0.06 - avg: 0.05931463465094566 - var: 2.5903962068696273e-06
r0: 0.07 - avg: 0.06973534822463989 - var: 2.2733129299012944e-06
r0: 0.08 - avg: 0.07968365401029587 - var: 2.446368398523191e-06
r0: 0.09 - avg: 0.09036508947610855 - var: 2.4145151655829977e-06
r0: 0.1 - avg: 0.10043637454509735 - var: 2.6910679480351973e-06
r0: 0.11 - avg: 0.10986917465925217 - var: 2.6786367470776895e-06
r0: 0.12 - avg: 0.11904612183570862 - var: 3.179570740030613e-06
r0: 0.13 - avg: 0.12874583899974823 - var: 3.7740276184194954e-06
r0: 0.14 - avg: 0.13885025680065155 - var: 4.445868398761377e-06
r0: 0.15 - avg: 0.14903776347637177 - var: 5.3434732762980275e-06
r0: 0.16 - avg: 0.1592549830675125 - var: 6.3222528297046665e-06
r0: 0.17 - avg: 0.1697094589471817 - var: 7.656590241822414e-06
r0: 0.18 - avg: 0.1802009493112564 - var: 8.129758498398587e-06
r0: 0.19 - avg: 0.189994677901268 - var: 7.782503416819964e-06
r0: 0.2 - avg: 0.19861596822738647 - var: 5.891980435990263e-06
{'0.001': [56.29, 44.62, 49.07, 49.0, 47.71, 45.47, 46.68, 37.33, 32.39, 31.94, 30.66, 29.25, 27.94, 27.16, 27.76, 29.01],
 '0.002': [88.95, 75.82, 81.4, 80.3, 79.17, 76.13, 78.44, 67.55, 60.57, 59.18, 57.24, 55.61, 53.12, 51.57, 52.11, 53.81],
 '0.005': [99.97, 99.19, 99.84, 99.61, 99.75, 99.57, 99.66, 98.79, 97.16, 96.42, 95.53, 94.25, 92.85, 91.94, 92.81, 92.4],
 '0.01': [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 99.99, 99.98, 99.96, 100.0, 99.96, 99.93]}

1906 (open)
Average Loss: 0.003010103179048747
r0: 0.05 - avg: 0.053755760192871094 - var: 1.4424102801058325e-06
r0: 0.06 - avg: 0.05963364243507385 - var: 1.2567466910695657e-05
r0: 0.07 - avg: 0.06984749436378479 - var: 7.314950380532537e-06
r0: 0.08 - avg: 0.07184547185897827 - var: 2.0652201783377677e-05
r0: 0.09 - avg: 0.08998594433069229 - var: 4.123939106648322e-06
r0: 0.1 - avg: 0.0997324213385582 - var: 4.658990292227827e-06
r0: 0.11 - avg: 0.10927006602287292 - var: 6.600593223993201e-06
r0: 0.12 - avg: 0.11911977827548981 - var: 8.36619710753439e-06
r0: 0.13 - avg: 0.12893332540988922 - var: 9.788422175915912e-06
r0: 0.14 - avg: 0.1395544856786728 - var: 9.803556167753413e-06
r0: 0.15 - avg: 0.14982910454273224 - var: 1.1151522812724579e-05
r0: 0.16 - avg: 0.160221129655838 - var: 1.2625013368960936e-05
r0: 0.17 - avg: 0.1708713322877884 - var: 1.4187800843501464e-05
r0: 0.18 - avg: 0.1824159175157547 - var: 1.2591990525834262e-05
r0: 0.19 - avg: 0.19299988448619843 - var: 1.1374634595995303e-05
r0: 0.2 - avg: 0.20194268226623535 - var: 7.387518053292297e-06
{'0.001': [0.76, 30.13, 29.24, 6.79, 38.34, 35.86, 31.09, 28.62, 26.2, 27.03, 25.19, 23.21, 20.53, 16.06, 14.26, 19.96],
 '0.002': [6.8, 54.28, 55.18, 12.17, 68.13, 64.18, 57.11, 52.67, 49.59, 50.86, 47.69, 44.67, 38.83, 32.57, 28.36, 40.36], 
 '0.005': [84.69, 88.97, 93.94, 17.94, 98.45, 97.93, 93.61, 90.68, 88.06, 89.66, 87.91, 84.89, 80.45, 74.02, 69.7, 86.69],
  '0.01': [100.0, 96.77, 99.75, 61.56, 99.99, 99.98, 99.86, 99.49, 99.17, 99.24, 99.16, 99.05, 99.12, 99.05, 98.93, 99.95]}


Average Loss: 0.0016211806989973412
r0: 0.05 - avg: 0.051297035068273544 - var: 1.332604483650357e-06
r0: 0.06 - avg: 0.060676515102386475 - var: 1.908332478706143e-06
r0: 0.07 - avg: 0.07044843584299088 - var: 1.9542767404345796e-06
r0: 0.08 - avg: 0.07990199327468872 - var: 1.98229395209637e-06
r0: 0.11 - avg: 0.10964389890432358 - var: 2.77444041785202e-06
r0: 0.12 - avg: 0.11975124478340149 - var: 3.170621312165167e-06
r0: 0.13 - avg: 0.1300373077392578 - var: 3.6478627407632302e-06
r0: 0.14 - avg: 0.14025495946407318 - var: 3.992317033407744e-06
r0: 0.15 - avg: 0.1504710465669632 - var: 4.4174676077091135e-06
r0: 0.16 - avg: 0.16080403327941895 - var: 5.081887138658203e-06
r0: 0.17 - avg: 0.1713593453168869 - var: 5.694986612070352e-06
r0: 0.18 - avg: 0.18186479806900024 - var: 6.253122592170257e-06
r0: 0.19 - avg: 0.19188684225082397 - var: 5.911998869123636e-06
r0: 0.2 - avg: 0.2004615068435669 - var: 4.719258413388161e-06
{'0.001': [37.13, 47.14, 49.81, 52.24, 50.14, 48.98, 44.51, 42.49, 40.57, 38.25, 36.26, 32.36, 27.66, 24.55, 22.82, 35.03],
 '0.002': [73.0, 80.08, 82.73, 84.68, 81.95, 80.7, 76.61, 73.52, 70.6, 68.28, 64.69, 59.61, 52.42, 46.14, 45.61, 62.96],
 '0.005': [99.87, 99.97, 99.96, 99.88, 99.88, 99.87, 99.6, 99.4, 99.02, 98.65, 97.87, 96.29, 93.26, 89.33, 89.86, 97.43],
 '0.01': [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 99.91, 99.96, 100.0]}
'''


