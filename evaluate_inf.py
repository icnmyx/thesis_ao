import numpy as np
import closed_loop_r0 as clr
from docopt import docopt
import os

from shesha.config import ParamConfig
from shesha.supervisor.compassSupervisor import CompassSupervisor as Supervisor
import matplotlib.pyplot as plt
import random
from models import Model, Model2, ModelA, ModelAuto, ModelA3, ModelAuto1
import torch
from collections import deque

"""
script to test a model against live simulation data.

"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dir = os.path.join(os.path.dirname(__file__), "data")

container = np.load(os.path.join(dir, f'wfs_data_closed_r0.05_1202_10000.npz'))
img = container['arr_0'][0]
img = [img / (1124599.2 * 1.1)]
img = torch.tensor(img).to(device)

# Model parameters
best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), 'best_model_0309_2.pth')
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), 'best_model_2207_2.pth')
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), 'best_model_1009_1.pth')

net = ModelAuto()
#net = ModelA3()
#net = ModelAuto1()

net.load_state_dict(torch.load(best_path))
net.to(device)
net.eval()

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
reps = 300
timings=np.zeros((reps,1))
# warm up gpu
for _ in range(10):
    _ = net(img)

# measure inference performance
for rep in range(reps):
    starter.record()
    _ = net(img)
    ender.record()

    # wait for gpu sync
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    timings[rep] = curr_time

mean_syn = np.sum(timings) / reps
std_syn = np.std(timings)
print(mean_syn)