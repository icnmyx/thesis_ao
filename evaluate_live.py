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
avg_losses = []
var_losses = []

# Model parameters
best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), 'best_model_0309_2.pth')
net = ModelAuto()

net.load_state_dict(torch.load(best_path))
net.to(device)
net.eval()
print(net, file=f)


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

    raw_predictions = []
    for i in range(samples):
      raw_predictions.append(generate_next())
    
    loss = [i - r0 for i in raw_predictions]
    avg_pred = np.mean(raw_predictions)
    variance = np.var(raw_predictions)

    avg_losses.append(np.mean(loss))
    var_losses.append(np.var(loss))

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

  print(f'Mean Error: {np.mean(avg_losses)}', file=f)

  # Calculate upper and lower bounds for the error bars (standard deviation)
  stdevs = np.sqrt(var_losses)
  lower_bounds = [avg - std_dev for avg, std_dev in zip(avg_losses, stdevs)]
  upper_bounds = [avg + std_dev for avg, std_dev in zip(avg_losses, stdevs)]

  # Create a bar chart with error bars
  bar_width = 0.35
  index = np.arange(len(turbs))

  fig, ax = plt.subplots(figsize=(10, 6))
  #bar1 = ax.bar(index, avg_losses, bar_width, label='Mean', alpha=0.6, color='b', yerr=stdevs, capsize=5)
  ax.errorbar(index, avg_losses, yerr=stdevs, fmt='o', markersize=6, color='black', label='Standard Deviation', elinewidth=1, capsize=10, capthick=1)

  # Add labels and title
  ax.set_xlabel('True r0', fontsize=14)
  ax.set_ylabel('Deviance from True r0', fontsize=14)
  ax.set_title('Distribution of r0 Predictions by True r0', fontsize=16)
  ax.set_xticks(index)
  ax.set_xticklabels(turbs)
  ax.grid(axis='y', linestyle='--', alpha=0.8)
  
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
  

'''
  # -- keep generating and updating
  while True:
    user_input = input("Press enter (or 'exit' to quit): ")

    if user_input == "exit":
        print("Exiting the program...")
        break

    # Call a function for any input other than "exit"
    ma = generate_next(True)

    #print(prediction_buffer)
'''

#0206 (10)
#Prediction Average: 0.08893897780030965 - Variance: 2.988311700111215e-06
#MA Average: 0.0889374320447445 - Variance: 1.93469284986792e-06

#1506 (10)
#Prediction Average: 0.1381690441966057 - Variance: 5.2949341331342614e-06
#MA Average: 0.13817668316215276 - Variance: 4.01600208881944e-06

#1506 (100)
#Prediction Average: 0.12930961626023055 - Variance: 4.162315861173637e-06
#MA Average: 0.1293447385075688 - Variance: 1.323956198200231e-06

#Prediction Average: 0.11962202878296375 - Variance: 4.544276826758281e-06
#MA Average: 0.11969999078921974 - Variance: 2.553969552114326e-06


