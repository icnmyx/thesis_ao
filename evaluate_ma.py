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
script to test a model against live simulation data with moving average.

"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# r0 params
r_min = 0.05
r_max = 0.2
open_loop = False
param_file = 'scao_sh_16x16_8pix.py'

turbs = np.arange(r_min, r_max, 0.01, dtype=float)
turbs = [round(x, 2) for x in turbs]

buffer_size = 100

best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '26_best_model1.pth')
net = ModelAuto()
net.load_state_dict(torch.load(best_path))
net.to(device)
net.eval()

dir = os.path.join(os.path.dirname(__file__), 'results')
f = open(os.path.join(dir, f'eval_ma_26_100.log'),'w')

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

    wfs_images = deque(maxlen=buffer_size) # -- save images and predictions
    prediction_buffer = deque(maxlen=buffer_size)
    moving_average = []

    def generate_next(verbose=False):
      wfs_im = sup.wfs.get_wfs_image(0) # -- get and save image
      wfs_images.append(wfs_im)
      #wfs_im = [wfs_im / (max(1124599.2, np.max(wfs_images)) * 1.1)]
      wfs_im = [wfs_im / (4361.1997 * 1.1)]
      #wfs_im = [wfs_im / (np.max(wfs_im)) * 1.1]

      wfs_im = torch.tensor(wfs_im).to(device)
      prediction = net(wfs_im).item()

      prediction_buffer.append(prediction)
      sup.loop(1)

      ma = np.mean(prediction_buffer)
      moving_average.append(ma)
      
      if verbose: 
        print(f"real r0: {r0} - newly predicted r0: {prediction:.4f}")
        print(f"updated moving_average: {ma:.4f} - error: {np.abs(ma - r0):.4f}")

      return ma, prediction

    # -- fill buffer
    for _ in range(buffer_size):
      ma, _ = generate_next()

    print(f"r0: {r0:.2f} - buffer_size: {buffer_size}")
    print(f"ma_prediction: {ma:.4f} - error: {np.abs(ma - r0):.4f}")

    raw_predictions = []
    mas = []
    for i in range(10000):
      ma, prediction = generate_next()
      mas.append(ma)
      raw_predictions.append(prediction)
    
    print(f'r0: {r0}', file=f)
    print(f'Prediction - Mean: {np.mean(raw_predictions)} - Variance: {np.var(raw_predictions)}', file=f)
    print(f'MA - Mean: {np.mean(mas)} - Variance: {np.var(mas)}', file=f)

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


