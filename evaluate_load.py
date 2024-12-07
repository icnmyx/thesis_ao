import numpy as np
import closed_loop_r0 as clr
from docopt import docopt
import os

from shesha.config import ParamConfig
from shesha.supervisor.compassSupervisor import CompassSupervisor as Supervisor
import matplotlib.pyplot as plt
import random
from models import Model, Model2, ModelA, ModelAuto, ModelA3, ModelAuto1, ModelAutoP
import torch
from collections import deque

"""
script to test a model against live simulation data.


1. toggle test data
2. select model
3. change normalisation
4. change both file names
"""
samples = 10000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

savepath = os.path.join(os.path.dirname(__file__), 'results')

f = open(os.path.join(savepath, f'eval_load_33_1_BENCH_13_{samples}.log'),'w')
#"eval_live_2109_3_BENCH_c2o.png"

file_name = f"eval_load_33_1_BENCH_13_{samples}.png"

# r0 params
r_min = 0.05
r_max = 0.2
turbs = np.arange(r_min, r_max, 0.01, dtype=float)
turbs = [round(x, 2) for x in turbs]

thresholds = [0.001, 0.002, 0.005, 0.01]
res = {}
avg_deviance = []
var_deviance = []
abs_errs = []

errs = []

dir = os.path.join(os.path.dirname(__file__), "data")

#3c 10000
#codes = ['r0.05_6764', 'r0.06_7620', 'r0.07_3215', 'r0.08_6519', 'r0.09_1466', 'r0.1_2619', 'r0.11_7235', 'r0.12_1380', 'r0.13_9753', 'r0.14_4506', 'r0.15_5935', 'r0.16_9415', 'r0.17_276', 'r0.18_7968', 'r0.19_4094', 'r0.2_3631']
#3c 30000
#codes = ['r0.05_9088', 'r0.06_2896', 'r0.07_3475', 'r0.08_5007', 'r0.09_2536', 'r0.1_1009', 'r0.11_9089', 'r0.12_9403', 'r0.13_4411', 'r0.14_5318', 'r0.15_4192', 'r0.16_5639', 'r0.17_3867', 'r0.18_7244', 'r0.19_5724', 'r0.2_6887']
#3o 10000
#codes = ['r0.05_2091', 'r0.06_2716', 'r0.07_2467', 'r0.08_5161', 'r0.09_5658', 'r0.1_6414', 'r0.11_568', 'r0.12_6749', 'r0.13_5478', 'r0.14_7598', 'r0.15_8168', 'r0.16_8029', 'r0.17_4701', 'r0.18_4740', 'r0.19_8281', 'r0.2_6153']
#9c 10000
#codes = ['r0.05_4028', 'r0.06_122', 'r0.07_8281', 'r0.08_7300', 'r0.09_9280', 'r0.1_6473', 'r0.11_7609', 'r0.12_4024', 'r0.13_6865', 'r0.14_3543', 'r0.15_1102', 'r0.16_8708', 'r0.17_7723', 'r0.18_3591', 'r0.19_5469', 'r0.2_84']
#10c 10000
#codes = ['r0.05_3778', 'r0.06_2857', 'r0.07_1407', 'r0.08_4864', 'r0.09_1798', 'r0.1_5039', 'r0.11_5368', 'r0.12_6839', 'r0.13_4478', 'r0.14_9872', 'r0.15_1118', 'r0.16_1674', 'r0.17_335', 'r0.18_1760', 'r0.19_9556', 'r0.2_5227']
#11c 10000
#codes = ['r0.05_5407', 'r0.06_7606', 'r0.07_3191', 'r0.08_918', 'r0.09_413', 'r0.1_2999', 'r0.11_6725', 'r0.12_4788', 'r0.13_2235', 'r0.14_9242', 'r0.15_2340', 'r0.16_8644', 'r0.17_9544', 'r0.18_8305', 'r0.19_8649', 'r0.2_1933']
#12c 10000
#codes = ['r0.05_8337', 'r0.06_9967', 'r0.07_7000', 'r0.08_546', 'r0.09_4167', 'r0.1_447', 'r0.11_9773', 'r0.12_2096', 'r0.13_9079', 'r0.14_4215', 'r0.15_8411', 'r0.16_1610', 'r0.17_4094', 'r0.18_8937', 'r0.19_8380', 'r0.2_3993']
#13c 10000
#codes = ['r0.05_2081', 'r0.06_5238', 'r0.07_4516', 'r0.08_6133', 'r0.09_671', 'r0.1_9538', 'r0.11_3884', 'r0.12_8612', 'r0.13_5994', 'r0.14_9114', 'r0.15_2188', 'r0.16_5777', 'r0.17_7022', 'r0.18_9084', 'r0.19_6159', 'r0.2_564']

#12c_0 10000
#codes = ['r0.05_827', 'r0.06_3408', 'r0.07_9711', 'r0.08_9292', 'r0.09_4467', 'r0.1_2675', 'r0.11_7578', 'r0.12_7249', 'r0.13_431', 'r0.14_257', 'r0.15_4428', 'r0.16_9952', 'r0.17_4354', 'r0.18_5497', 'r0.19_6454', 'r0.2_1070']
#12c_1 10000
#codes = ['r0.05_999', 'r0.06_6269', 'r0.07_8685', 'r0.08_5091', 'r0.09_5349', 'r0.1_8852', 'r0.11_2565', 'r0.12_4889', 'r0.13_8675', 'r0.14_5839', 'r0.15_7902', 'r0.16_8959', 'r0.17_9738', 'r0.18_735', 'r0.19_1934', 'r0.2_2972']

#9c_1 10000
#codes = ['r0.05_6687', 'r0.06_8720', 'r0.07_2426', 'r0.08_67', 'r0.09_2665', 'r0.1_37', 'r0.11_7136', 'r0.12_7448', 'r0.13_9', 'r0.14_5809', 'r0.15_8055', 'r0.16_5180', 'r0.17_141', 'r0.18_9878', 'r0.19_4613', 'r0.2_2105']
#10c_1 10000
#codes = ['r0.05_724', 'r0.06_5594', 'r0.07_4394', 'r0.08_3785', 'r0.09_1006', 'r0.1_8159', 'r0.11_5608', 'r0.12_9310', 'r0.13_9410', 'r0.14_4052', 'r0.15_6102', 'r0.16_7689', 'r0.17_1811', 'r0.18_1796', 'r0.19_7005', 'r0.2_9750']
#11c_1 10000
#codes = ['r0.05_9799', 'r0.06_2987', 'r0.07_7673', 'r0.08_9671', 'r0.09_9934', 'r0.1_116', 'r0.11_4312', 'r0.12_3683', 'r0.13_9274', 'r0.14_2269', 'r0.15_2791', 'r0.16_6301', 'r0.17_9234', 'r0.18_9629', 'r0.19_1964', 'r0.2_9314']
#13c_1 10000
#codes = ['r0.05_1035', 'r0.06_5236', 'r0.07_3121', 'r0.08_7452', 'r0.09_4851', 'r0.1_7444', 'r0.11_3322', 'r0.12_7805', 'r0.13_7150', 'r0.14_8301', 'r0.15_8487', 'r0.16_4078', 'r0.17_5203', 'r0.18_7680', 'r0.19_531', 'r0.2_5055']

# PYR
# 3c 10000
#codes = ['r0.05_5901', 'r0.06_8645', 'r0.07_9350', 'r0.08_6254', 'r0.09_3919', 'r0.1_7765', 'r0.11_6034', 'r0.12_3561', 'r0.13_6879', 'r0.14_4492', 'r0.15_8935', 'r0.16_8678', 'r0.17_4982', 'r0.18_9146', 'r0.19_2060', 'r0.2_7795']
# 3o 10000
#codes = ['r0.05_7048', 'r0.06_7068', 'r0.07_5178', 'r0.08_9813', 'r0.09_9995', 'r0.1_3278', 'r0.11_3105', 'r0.12_5476', 'r0.13_6554', 'r0.14_8208', 'r0.15_8619', 'r0.16_5507', 'r0.17_2443', 'r0.18_9548', 'r0.19_5275', 'r0.2_701']
# 9c_1 10000 
#codes = ['r0.05_7158', 'r0.06_7417', 'r0.07_3817', 'r0.08_5833', 'r0.09_1608', 'r0.1_4218', 'r0.11_6226', 'r0.12_122', 'r0.13_8798', 'r0.14_4942', 'r0.15_1196', 'r0.16_5979', 'r0.17_7173', 'r0.18_4289', 'r0.19_8225', 'r0.2_4845']
# 10c_1 10000
#codes = ['r0.05_1681', 'r0.06_614', 'r0.07_1400', 'r0.08_4320', 'r0.09_1330', 'r0.1_312', 'r0.11_6766', 'r0.12_831', 'r0.13_9239', 'r0.14_7114', 'r0.15_2452', 'r0.16_9693', 'r0.17_2380', 'r0.18_5877', 'r0.19_5497', 'r0.2_3110']
# 11c_1 10000
#codes = ['r0.05_7386', 'r0.06_9444', 'r0.07_2439', 'r0.08_7383', 'r0.09_3492', 'r0.1_1539', 'r0.11_2146', 'r0.12_9557', 'r0.13_5783', 'r0.14_3383', 'r0.15_7151', 'r0.16_6498', 'r0.17_9684', 'r0.18_9710', 'r0.19_6350', 'r0.2_5959']
# 12c_1 10000 #pyr_12_1_closed
#codes = ['r0.05_1781', 'r0.06_2165', 'r0.07_1863', 'r0.08_2668', 'r0.09_9374', 'r0.1_6246', 'r0.11_5897', 'r0.12_2765', 'r0.13_5187', 'r0.14_3193', 'r0.15_4266', 'r0.16_6487', 'r0.17_4310', 'r0.18_8205', 'r0.19_608','r0.2_5628']
# 13c_1 10000
codes = ['r0.05_1330', 'r0.06_6572', 'r0.07_7577', 'r0.08_4757', 'r0.09_29', 'r0.1_730', 'r0.11_3778', 'r0.12_76', 'r0.13_2114', 'r0.14_5309', 'r0.15_745', 'r0.16_9779', 'r0.17_7444', 'r0.18_6214', 'r0.19_3793', 'r0.2_850']



# Model parameters
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), 'best_model_2109_3.pth') # bench closed
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), 'best_model_2209_2.pth') # bench open
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), 'best_model_2309_1.pth') # bench 3co
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '19_best_model2.pth') # bench 3co half (1162616.9)
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '21_best_model1.pth') # bench closed (open norm)
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '22_best_model1.pth') # GSMAG 9 (4416.9136)
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '23_best_model1.pth') # GSMAG 9-11 (4338.618)
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '24_best_model1.pth') # Noise 12(-1) (277.07904)
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '25_best_model2.pth') # Noise 12(-1 0 1) (308.80664)
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '26_best_model1.pth') # 9-11 (1) (4361.1997)


#pyr
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '29_best_model2.pth') # 3c (2099258.8)
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '28_best_model2.pth') # 3o (2383758.0)
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '30_best_model2.pth') # 3co (2383758.0)
#best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '31_best_model2.pth') # 3c open norm (2383758.0)
best_path = os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), '33_best_model1.pth') # 9-11 1 (6957.9688)


net = ModelAutoP()
#net = ModelAuto()

net.load_state_dict(torch.load(best_path))
net.to(device)
net.eval()
print(net, file=f)

for (i, r0) in enumerate(turbs):

  path = os.path.join(dir, f'eval_pyr_13_1_closed_{codes[i]}_{samples}.npz')
  container = np.load(path)
  img_data = container['arr_0']
  label_data = container['arr_1']

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
  X = torch.Tensor(np.array(img_data) / (6957.9688 * 1.1))
  y = torch.Tensor([np.round(ld, 2) for ld in label_data])

  X = X.to(device)
  raw_predictions = net(X).tolist()
  #print(type(raw_predictions))
  #print(np.mean(raw_predictions))
    
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

  errs = [*errs, *[abs(i) for i in deviance]]

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
  #print(f'dev mean: {np.mean(deviance)} - var: {np.var(deviance)}', file=f)
  
mae = np.mean(abs_errs)
mse = np.mean([i ** 2 for i in abs_errs])

ssr = np.sum([(i - mae) ** 2 for i in abs_errs])
sst = np.sum([samples * ((turbs - np.mean(turbs)) ** 2) for i in turbs])
r2 = 1 - (ssr / sst)

print(f'Mean Absolute Error: {mae}', file=f)
#print(f'Mean Squared Error: {mse}', file=f)
print(f'Root Mean Squared Error: {np.sqrt(mse)}', file=f)
print(f'r2: {r2}', file=f)

print(f'dev mean: {avg_deviance}', file=f)
print(f'dev var: {var_deviance}', file=f)

print('', file=f)
mae = np.mean(errs)
print(f'mae: {mae}', file=f)
print(f'var: {np.var(errs)}', file=f)
mse = np.mean([err ** 2 for err in errs])
rmse = np.sqrt(mse)
print(f'rmse: {rmse}', file=f)
  
#print(len(errs))
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
ax.set_xlabel('True r0 (m)', fontsize=14)
ax.set_ylabel('Deviance from True r0 (m)', fontsize=14)
#ax.set_title('Distribution of r0 Predictions by True r0', fontsize=16)
ax.set_xticks(index)
ax.set_xticklabels(turbs)
ax.grid(axis='y', linestyle='--', alpha=0.8)

# Add a thicker line at y=0 on the y-axis
ax.axhline(y=0, color='k', linestyle='--', linewidth=1, label='Zero Line', alpha=0.8)

# Show the plot
plt.ylim(min(lower_bounds) - 0.002, max(upper_bounds) + 0.002)
ax.set_aspect('auto')
plt.tight_layout()

plt.savefig(file_name)
plt.show()

  

