import numpy as np
import closed_loop_r0 as clr
from docopt import docopt
import os

from shesha.config import ParamConfig
from shesha.supervisor.compassSupervisor import CompassSupervisor as Supervisor
import matplotlib.pyplot as plt
import random
from models import Model, Model2, ModelA, ModelAuto
import torch
from collections import deque

"""
script to test a model against live simulation data.

"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# r0 params
r_min = 0.05
r_max = 0.2
open_loop = False
param_file = 'scao_sh_16x16_8pix_noise.py'

turbs = np.arange(r_min, r_max, 0.01, dtype=float)
turbs = [round(x, 2) for x in turbs]
turbs = [0.05, 0.1, 0.2]

buffer_size = 1000

best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '26_best_model1.pth')
net = ModelAuto()
net.load_state_dict(torch.load(best_path))
net.to(device)
net.eval()

dir = os.path.join(os.path.dirname(__file__), 'results')
f = open(os.path.join(dir, f'eval_mas_26_100.log'),'w')

if __name__ == "__main__":
  #colors = ['#ffb95d', '#fe8260', '#e65ca0', '#b660ca', '#8e63dd']
  #colors=['#FF6B35', '#2A9D8F', '#E9C46A', '#264653']
  colors=['#ffb95d', '#fe8260', '#457B9D', '#264653']
  raw = []
  ma_10 = []
  ma_100 = []
  ma_1000 = []
  for r0 in turbs:
    
    # instantiates class and load parameters
    config = ParamConfig(param_file)
    config.p_atmos.set_seeds([random.randint(0,9999)]) # set random 4-digit seed

    sup = Supervisor(config, silence_tqdm=True) 

    #r0 = round(random.uniform(r_min, r_max), 2)
    #r0 = 0.1

    sup.atmos.set_r0(float(r0)) # -- set r0 for simulation
    sup.loop(1000) # -- let the change take effect

    # set open loop
    if open_loop:
      sup.rtc.open_loop()

    wfs_images = deque(maxlen=buffer_size) # -- save images and predictions
    prediction_buffer_10 = deque(maxlen=10)
    prediction_buffer_100 = deque(maxlen=100)
    prediction_buffer_1000 = deque(maxlen=1000)
    moving_average_1 = []
    moving_average_10 = []
    moving_average_100 = []
    moving_average_1000 = []

    var_1 = []
    var_10 = []
    var_100 = []
    var_1000 = []

    def generate_next():
      wfs_im = sup.wfs.get_wfs_image(0) # -- get and save image
      wfs_images.append(wfs_im)
      #wfs_im = [wfs_im / (max(1124599.2, np.max(wfs_images)) * 1.1)]
      wfs_im = [wfs_im / (4361.1997 * 1.1)]
      #wfs_im = [wfs_im / (np.max(wfs_im)) * 1.1]

      wfs_im = torch.tensor(wfs_im).to(device)
      prediction = net(wfs_im).item()

      sup.loop(1)

      return prediction

    # -- fill buffer
    for _ in range(2000):
      prediction = generate_next()

      prediction_buffer_10.append(prediction)
      prediction_buffer_100.append(prediction)
      prediction_buffer_1000.append(prediction)

      #moving_average_1.append(prediction)
      #moving_average_10.append(np.mean(prediction_buffer_10))
      #moving_average_100.append(np.mean(prediction_buffer_100))
      #moving_average_1000.append(np.mean(prediction_buffer_1000))

      #var_1.append(np.var(moving_average_1))
      #var_10.append(np.var(moving_average_10))
      #var_100.append(np.var(moving_average_100))
      #var_1000.append(np.var(moving_average_1000))

      #moving_average_10.append(np.mean(prediction_buffer_10))
      #moving_average_100.append(np.mean(prediction_buffer_100))
      #moving_average_1000.append(np.mean(prediction_buffer_1000))
    '''
    if r0 in [0.05, 0.1, 0.2]:
      # Create a fancy plot
      plt.figure(figsize=(10, 6))
      plt.title(f'Variance Across Different Moving Average Sizes @ r0 = {r0}', fontsize=16, pad=20)
      plt.xlabel('Frames', fontsize=14)
      plt.ylabel('Variance', fontsize=14)
      plt.grid(True, linestyle='--', alpha=0.7)

      # Plot each series with different colors and styles
      
      linestyles = ['-', '--', '-.', ':']

      plt.plot(var_1, label='Prediction Variance', linestyle=linestyles[0], color = colors[0])
      plt.plot(var_10, label='MA10 Variance', linestyle=linestyles[1], color = colors[1])
      plt.plot(var_100, label='MA100 Variance', linestyle=linestyles[2], color = colors[2])
      plt.plot(var_1000, label='MA1000 Variance', linestyle=linestyles[3], color = colors[3])

      # Add a legend
      plt.legend(loc='upper right')

      # Save or display the plot
      plt.tight_layout()
      plt.savefig(f'variance_comparison_{r0}.png', dpi=300)  # Save the plot to a file
      plt.show()  # Display the plot
    '''

    #print(f"r0: {r0:.2f} - buffer_size: {buffer_size}")
    #print(f"ma_prediction: {ma:.4f} - error: {np.abs(ma - r0):.4f}")

    moving_average_10 = []
    moving_average_100 = []
    moving_average_1000 = []
    raw_predictions = []
    mas = []
    for i in range(10000):
      prediction = generate_next()

      prediction_buffer_10.append(prediction)
      prediction_buffer_100.append(prediction)
      prediction_buffer_1000.append(prediction)

      moving_average_10.append(np.mean(prediction_buffer_10))
      moving_average_100.append(np.mean(prediction_buffer_100))
      moving_average_1000.append(np.mean(prediction_buffer_1000))
      
      raw_predictions.append(prediction)

      var_1.append(np.var(raw_predictions))
      var_10.append(np.var(moving_average_10))
      var_100.append(np.var(moving_average_100))
      var_1000.append(np.var(moving_average_1000))
    
    print(f'r0: {r0}', file=f)
    print(f'Prediction - Mean: {np.mean(raw_predictions)} - Variance: {np.var(raw_predictions)}', file=f)
    print(f'MA10 - Mean: {np.mean(moving_average_10)} - Variance: {np.var(moving_average_10)}', file=f)
    print(f'MA100 - Mean: {np.mean(moving_average_100)} - Variance: {np.var(moving_average_100)}', file=f)
    print(f'MA1000 - Mean: {np.mean(moving_average_1000)} - Variance: {np.var(moving_average_1000)}', file=f)

    raw.append((np.mean(raw_predictions), np.var(raw_predictions)))
    ma_10.append((np.mean(moving_average_10), np.var(moving_average_10)))
    ma_100.append((np.mean(moving_average_100), np.var(moving_average_100)))
    ma_1000.append((np.mean(moving_average_1000), np.var(moving_average_1000)))


    if r0 in [0.05, 0.1, 0.2]:

      # Create an array of x-values (assuming equal spacing between data points)
      x_values = np.arange(len(raw_predictions))

      # Create the plot
      plt.figure(figsize=(10, 6))
      plt.plot(x_values, raw_predictions, label='Raw Estimations', color = colors[0])
      plt.plot(x_values, moving_average_10, label=f'MA{10}', color = colors[1])
      plt.plot(x_values, moving_average_100, label=f'MA{100}', color = colors[2])
      plt.plot(x_values, moving_average_1000, label=f'MA{1000}', color = colors[3])

      # Customize the plot
      plt.title(f'Raw Estimations and Moving Averages @ r0 = {r0}', fontsize=16, pad=20)
      plt.xlabel('Frames', fontsize=14)
      plt.ylabel('Fried Parameter Estimation', fontsize=14)
      plt.legend()
      plt.grid(True, linestyle='--', alpha=0.7)

      # Show the plot
      plt.tight_layout()
      plt.show()
      plt.savefig(f'mas_{r0}_10000.png', dpi=300)


    if r0 in [0.05, 0.1, 0.2]:
      # Create a fancy plot
      plt.figure(figsize=(10, 6))
      plt.title(f'Running Variance Across Different Moving Average Sizes @ r0 = {r0}', fontsize=16, pad=20)
      plt.xlabel('Frames', fontsize=14)
      plt.ylabel('Variance', fontsize=14)
      plt.grid(True, linestyle='--', alpha=0.7)

      # Plot each series with different colors and styles
      
      linestyles = ['-', '--', '-.', ':']

      plt.plot(var_1, label='Prediction Variance', linestyle=linestyles[0], color = colors[0])
      plt.plot(var_10, label='MA10 Variance', linestyle=linestyles[1], color = colors[1])
      plt.plot(var_100, label='MA100 Variance', linestyle=linestyles[2], color = colors[2])
      plt.plot(var_1000, label='MA1000 Variance', linestyle=linestyles[3], color = colors[3])

      # Add a legend
      plt.legend()

      # Save or display the plot
      plt.tight_layout()
      plt.savefig(f'variance_comparison_{r0}_10000.png', dpi=300)  # Save the plot to a file
      plt.show()  # Display the plot

    #dir = os.path.join(os.path.dirname(__file__), 'results')
    #save_dir = os.path.join(dir, f'predictions_r{r0}.npz')
    #np.savez(save_dir, *raw_predictions)
    #save_dir = os.path.join(dir, f'ma_r{r0}.npz')
    #np.savez(save_dir, *mas)



