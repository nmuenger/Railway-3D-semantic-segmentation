import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import copy
import random
import time 
sys.path.insert(1, '/workspaces/baseline')
from pointcept.datasets.osdar23 import OSDaR23Dataset
from railseg.pcd_processing import pcd_to_las
from pointcept.datasets.transform import  Sparsify


annotation_disregarded = -2
learning_map = {
            'background':         0,
            'person':             1,
            'crowd':              1,
            'train':              2,
            'wagons':             2,
            'bicycle':            0,
            'group_of_bicycles':  0,
            'motorcycle':         0,
            'road_vehicle':       3,
            'animal':             0,
            'group_of_animals':   0,
            'wheelchair':         0,
            'drag_shoe':          0,
            'track':              4,
            'transition':         4,
            'switch':             annotation_disregarded,
            'catenary_pole':      5,
            'signal_pole':        6,
            'signal':             6,
            'signal_bridge':      0,
            'buffer_stop':        7,
            'flame':              0,
            'smoke':              0
        }



# ----------- Start of the code ------------
random.seed(32)

data_prev = OSDaR23Dataset(data_root="/workspaces/baseline/data/OSDaR_dataset/v_2", learning_map=learning_map, use_preprocessed=False)
data_new = OSDaR23Dataset(data_root="/workspaces/baseline/exp/preprocessed_pcd", learning_map=learning_map)
idx = 20
#rand_int1 = random.randint(0, 1048)
list_time_prev = []
list_time_new = []

for i in range(1049):
    
    time0 = time.time()
    data_dict_prev_method = data_prev.get_data(i) # was 140 before
    time1 = time.time()
    data_dict_new_method = data_new.get_data(i) # was 140 before
    time2 = time.time()
    time_elapsed_prev = time1-time0
    time_elapsed_new = time2-time1
    print(f"i={i}: Took {time_elapsed_prev} for original method, {time_elapsed_new} for new method.")
    list_time_prev.append(time_elapsed_prev)
    list_time_new.append(time_elapsed_new)


print(f"Total time for prev_method: {sum(list_time_prev)}, for new method: {sum(list_time_new)}")
