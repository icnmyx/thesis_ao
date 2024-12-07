import numpy as np
import closed_loop_r0 as clr
from docopt import docopt
import os

from shesha.config import ParamConfig
from shesha.supervisor.compassSupervisor import CompassSupervisor as Supervisor
import matplotlib.pyplot as plt
import random
from models import Model, Model2, ModelA
import torch
from collections import deque

"""
script to save images

"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# r0 params
r_min = 0.19
r_max = 0.2
open_loop = False
#param_file = 'scao_sh_16x16_8pix.py'
param_file = 'scao_pyrhr_16x16.py'


turbs = np.arange(r_min, r_max, 0.01, dtype=float)
turbs = [round(x, 2) for x in turbs]
print(turbs)
turbs = [0.05, 0.1, 0.2]

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

  wfs_im = sup.wfs.get_wfs_image(0)
  wfs_im = wfs_im / np.max(wfs_im)
  plt.gray()


  wfs_comp = np.split(wfs_im, 2, axis=0)
  wfs_comp = np.concatenate([np.split(c, 2, axis=1) for c in wfs_comp])

  #print(wfs_comp.shape)

  plt.imshow(wfs_im)
  #plt.axis('off')
  plt.savefig(f's_pyr_0_{r0}.png')
  
  '''
  counter = 0 
  for im in wfs_comp:
    counter += 1
    plt.imshow(im)
    plt.savefig(f's_pyr_{counter}.png')
  '''

    
  #zero_channel = np.zeros_like(wfs_im)
  #bluescale = np.stack([zero_channel, zero_channel, wfs_im], axis=2)

  #plt.imshow(wfs_im, cmap='gray')
  #plt.axis('off')
  #plt.savefig('sample.png')
  #print('done')


