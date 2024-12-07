import numpy as np
import closed_loop_r0 as clr
from docopt import docopt
import os

from shesha.config import ParamConfig
from shesha.supervisor.compassSupervisor import CompassSupervisor as Supervisor
import matplotlib.pyplot as plt
import random
from models import Model, Model2, ModelA, ModelAuto, ModelA3, ModelAuto1, ModelAutoS, ModelAutoP
import torch
from collections import deque

"""
script to test a model against live simulation data.

"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dir = os.path.join(os.path.dirname(__file__), "data")

#container = np.load(os.path.join(dir, f'wfs_data_closed_r0.05_1202_10000.npz'))
container = np.load(os.path.join(dir, f'pyr_9_1_closed_r0.1_8133_10000.npz'))
img = container['arr_0'][0]
img = [img / (6957.9688 * 1.1)]#1124599.2
#img = [img]
img = torch.tensor(img).to(device)

# Model parameters
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), 'best_model_0309_2.pth')
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), 'best_model_2207_2.pth')
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), 'best_model_1009_1.pth')
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), 'best_model_2209_2.pth')


#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), 'best_model_2109_3.pth') # bench closed 
#mean inference time: 0.8378 (ms) - std inference (ms): 0.0123 (ms)
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), 'best_model_2209_2.pth') # bench open
#mean inference time: 0.8246 (ms) - std inference (ms): 0.0100 (ms)
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), 'best_model_2309_1.pth') # bench 3co
#mean inference time: 0.8263 (ms) - std inference (ms): 0.0101 (ms)
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '19_best_model2.pth') # bench 3co half (1162616.9)
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '21_best_model1.pth') # bench closed (open norm)
#mean inference time: 0.8341 (ms) - std inference (ms): 0.0141 (ms)
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '22_best_model1.pth') # GSMAG 9 (4416.9136)
#mean inference time: 0.8299 (ms) - std inference (ms): 0.0100 (ms)
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '23_best_model1.pth') # GSMAG 9-11 (4338.618)
#mean inference time: 0.8244 (ms) - std inference (ms): 0.0103 (ms)
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '24_best_model1.pth') # Noise 12(-1) (277.07904)
#mean inference time: 0.8318 (ms) - std inference (ms): 0.0104 (ms)
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '25_best_model2.pth') # Noise 12(-1 0 1) (308.80664)
#mean inference time: 0.8344 (ms) - std inference (ms): 0.0096 (ms)
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '26_best_model1.pth') # 9-11 (1) (4361.1997)
#mean inference time: 0.8276 (ms) - std inference (ms): 0.0107 (ms)

#pyr
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '29_best_model2.pth') # 3c (2099258.8)
#mean inference time: 0.7772 (ms) - std inference (ms): 0.0296 (ms)
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '28_best_model2.pth') # 3o (2383758.0)
#mean inference time: 0.7679 (ms) - std inference (ms): 0.0314 (ms)
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '30_best_model2.pth') # 3co (2383758.0)
#mean inference time: 0.7750 (ms) - std inference (ms): 0.0307 (ms)
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '31_best_model2.pth') # 3c open norm (2383758.0)
#mean inference time: 0.7778 (ms) - std inference (ms): 0.0318 (ms)
best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '33_best_model1.pth') # 9-11 1 (6957.9688)


#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '18_best_model1.pth')

#net = ModelAuto() #- 72497
#net = ModelA3()
#net = ModelAuto1()
#net = ModelAutoS() #- 73265
net = ModelAutoP() #- 73697

net.load_state_dict(torch.load(best_path))
net.to(device)
net.eval()

total_params = sum(p.numel() for p in net.parameters())
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

#print(f"total params: {total_params} - trainable params: {trainable_params}")

#print(img.shape)
#img = torch.randn(1, 128, 128, dtype=torch.float).to(device)
img = torch.randn(16, 4, 32, 32, dtype=torch.float).to(device)

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
reps = 300
timings=np.zeros((reps,1))
# warm up gpu
for _ in range(300):  
    _ = net(img)
    img = img.squeeze(0)

# measure inference performance
for rep in range(reps):
    starter.record()
    _ = net(img)
    ender.record()

    # wait for gpu sync
    torch.cuda.synchronize()
    #Returns the time elapsed in milliseconds after the event was recorded and before the end_event was recorded.
    curr_time = starter.elapsed_time(ender) 
    timings[rep] = curr_time

    img = img.squeeze(0)

mean_syn = np.sum(timings) / reps
std_syn = np.std(timings)
print(f"mean inference time: {mean_syn:.4f} (ms) - std inference (ms): {std_syn:.4f} (ms)")

#mean inference time: 0.8229 (ms) - std inference (ms): 0.0094 (ms)