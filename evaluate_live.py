import numpy as np
import closed_loop_r0 as clr
from docopt import docopt
import os

from shesha.config import ParamConfig
from shesha.supervisor.compassSupervisor import CompassSupervisor as Supervisor
import matplotlib.pyplot as plt
import random
from models import Model, Model2, ModelA, ModelAuto, ModelA3
import torch
from collections import deque

"""
script to test a model against live simulation data.

"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dir = os.path.join(os.path.dirname(__file__), 'results')
f = open(os.path.join(dir, f'eval_live_1.log'),'w')

# r0 params
r_min = 0.05
r_max = 0.2
samples = 1000
open_loop = False
param_file = 'scao_sh_16x16_8pix.py'

turbs = np.arange(r_min, r_max, 0.01, dtype=float)
turbs = [round(x, 2) for x in turbs]

thresholds = [0.001, 0.002, 0.005, 0.01]
res = {}
avg_deviance = []
var_deviance = []
abs_errs = []

# Model parameters
best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), 'best_model_0309_2.pth')
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), 'best_model_2307_2.pth')
net = ModelAuto()
#net = ModelA3()

net.load_state_dict(torch.load(best_path))
net.to(device)
net.eval()
print(net, file=f)
rand_seed = random.randint(0,9999)
print(rand_seed)

if __name__ == "__main__":
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

    def generate_next(verbose=False):
      wfs_im = sup.wfs.get_wfs_image(0) # -- get and save image

      # Normlisation
      # 3 bench closed = 1124599.2
      # 3 bench open = 1162616.9
      # 10 = 1757.3783
      # 12 incl noise = 308.80664
      # 12 excl noise = 273.5929
      # 10-12 = 1757.3783
      # pyr 5 = 365248.0
      # pyr 12 excl noise = 471.5663
      # pyr 12 incl noise = 490.0
      # pyr 10-12 = 2726.0142
      # pyr 3 bench = 2099258.8
      wfs_im = [wfs_im / (1124599.2 * 1.1)]

      wfs_im = torch.tensor(wfs_im).to(device)
      prediction = net(wfs_im).item()
      sup.loop(1)
      
      if verbose: 
        print(f"real r0: {r0} - newly predicted r0: {prediction:.4f} - error: {np.abs(ma - r0):.4f}")

      return prediction

    # generate predictions on live data
    raw_predictions = []
    for i in range(samples):
      raw_predictions.append(generate_next())
    
    # infer statistics from predictions
    avg_pred = np.mean(raw_predictions)
    variance = np.var(raw_predictions)

    # calculate the deviance from true r0, then determine mean + var
    deviance = [i - r0 for i in raw_predictions]
    avg_deviance.append(np.mean(deviance))
    var_deviance.append(np.var(deviance))

    # calulate abs error for this r0
    abs_err = np.mean([abs(i) for i in deviance])
    abs_errs.append(abs_err)

    # Distribution
    dist = []
    for t in thresholds:
        lower = r0 - t
        upper = r0 + t
        in_range = np.sum([1 if (p >= lower) & (p <= upper) else 0 for p in raw_predictions])
        perc = (in_range / len(raw_predictions)) * 100
        dist.append(round(perc, 4))
    res[r0] = [dist]
    
    print(f'r0: {r0} - Mean: {avg_pred} - Variance: {variance} - Distribution: {dist}', file=f)

  print(f'Mean Absolute Error: {np.mean(abs_errs)}', file=f)

  # Calculate upper and lower bounds for the error bars (standard deviation)
  stdevs = np.sqrt(var_deviance)
  lower_bounds = [avg - std_dev for avg, std_dev in zip(avg_deviance, stdevs)]
  upper_bounds = [avg + std_dev for avg, std_dev in zip(avg_deviance, stdevs)]

  # Create a bar chart with error bars
  bar_width = 0.35
  index = np.arange(len(turbs))

  fig, ax = plt.subplots(figsize=(10, 6))
  #bar1 = ax.bar(index, avg_deviance, bar_width, label='Mean', alpha=0.6, color='b', yerr=stdevs, capsize=5)
  ax.errorbar(index, avg_deviance, yerr=stdevs, fmt='o', markersize=6, color='k', label='Standard Deviation', elinewidth=1, capsize=10, capthick=1)

  # Add labels and title
  ax.set_xlabel('True r0', fontsize=14)
  ax.set_ylabel('Deviance from True r0', fontsize=14)
  ax.set_title('Distribution of r0 Predictions by True r0', fontsize=16)
  ax.set_xticks(index)
  ax.set_xticklabels(turbs)
  ax.grid(axis='y', linestyle='--', alpha=0.8)

  # Add a thicker line at y=0 on the y-axis
  ax.axhline(y=0, color='k', linestyle='--', linewidth=1, label='Zero Line', alpha=0.8)

  # Show the plot
  plt.ylim(min(lower_bounds) - 0.002, max(upper_bounds) + 0.002)
  ax.set_aspect('auto')
  plt.tight_layout()

  file_name = "eval_live_1.png"
  plt.savefig(file_name)
  plt.show()

    #print([round(i, 2) for i in dist])
        #print(f"{perc:.2f}% of predictions within range {t}")
        #if str(t) in res:
        #  temp = res[str(t)]
        #  temp.append(round(perc, 2))
        #  res[str(t)] = temp
        #else:
        #  res[str(t)] = [round(perc, 2)]

    #dir = os.path.join(os.path.dirname(__file__), 'results')
    #save_dir = os.path.join(dir, f'predictions_r{r0}.npz')
    #np.savez(save_dir, *raw_predictions)
    #save_dir = os.path.join(dir, f'ma_r{r0}.npz')
    #np.savez(save_dir, *mas)
  

