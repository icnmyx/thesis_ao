import numpy as np
import closed_loop_r0 as clr
from docopt import docopt
import os

from shesha.config import ParamConfig
from shesha.supervisor.compassSupervisor import CompassSupervisor as Supervisor
import matplotlib.pyplot as plt
import random

"""
script generate a dataset of wfs images, with different r0 values

Usage:
  generate_dataset.py

"""

samples = 10000
r_min = 0.05
r_max = 0.2
r_int = 0.01
open_loop = False
turbs = np.arange(r_min, r_max, r_int, dtype=float)
turbs = [round(x, 2) for x in turbs]
savepath="data"

#param_file = 'scao_sh_16x16_8pix.py'
param_file = 'scao_sh_16x16_8pix_noise.py'

if __name__ == "__main__":

  # instantiates class and load parameters
  config = ParamConfig(param_file)
  

  # generate samples by iterating over r0 values
  for r0 in turbs:

    config.p_atmos.set_seeds([random.randint(0,9999)]) # set random 4-digit seed

    sup = Supervisor(config, silence_tqdm=True) 

    # set open loop
    if open_loop:
      sup.rtc.open_loop()
     
    wfs_images = [] # -- save images and labels 
    wfs_labels = []

    sup.atmos.set_r0(float(r0)) # -- set r0 for simulation
    sup.loop(1000) # -- let the change take effect
    
    # generate x many samples for each r0
    for i in range(samples):
      sup.loop(100)

      wfs_im = sup.wfs.get_wfs_image(0) # -- get and save image
      wfs_images.append(wfs_im)
      wfs_labels.append(r0)

    data = [wfs_images, wfs_labels]

    dir = os.path.join(os.path.dirname(__file__), savepath)
    save_dir = os.path.join(dir, f'wfs_12_closed_r{r0}_{str(config.p_atmos.get_seeds()[0])}_{len(wfs_images)}.npz')
    
    np.savez(save_dir, *data)
    