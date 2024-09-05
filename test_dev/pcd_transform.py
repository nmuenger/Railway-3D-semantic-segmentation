import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import copy
import random
import time 
from copy import deepcopy
sys.path.insert(1, '/workspaces/baseline')
from pointcept.datasets.osdar23 import OSDaR23Dataset
from railseg.pcd_processing import pcd_to_las
from pointcept.datasets.transform import PolarMixPaste, Sparsify
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

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

learning_map_inv = {
            #ignore_index: ignore_index,  # "unlabeled"
            0 : 'background',
            1 : 'person',
            2 : 'train' ,       
            3 : 'road_vehicle',
            4 : 'track',
            5 : 'catenary_pole',
            6 : 'signal',
            7 : 'buffer_stop',
        }

color_map = {#ignore_index:[211, 211, 211],
            0:[211, 211, 211],
            1: [255, 0, 0],  # Person -> Red
            2: [255, 215, 0],# Train -> Yellow
            3: [0, 139, 139],# Road vehicle -> Dark blue
            4: [255, 0, 255],# Track -> Pink
            5: [255, 140, 0],# Catenary_pole -> Orange
            6: [0, 191, 255],# Signal -> Flash blue
            7: [186, 85, 211],# Buffer stop -> Purple
            }

# ----------- Start of the code ------------
random.seed(32) # Good seed (and the one I made the exemple with): 32

data = OSDaR23Dataset(data_root="/workspaces/baseline/exp/preprocessed_pcd", learning_map=learning_map)
rand_int1 = random.randint(0, 1048)
rand_int2 = random.randint(0, 1048)

data_dict = data.get_data(rand_int1) # was 140 before
data_dict2 = data.get_data(rand_int2)

data_dict2 = {k+"_2": v for k, v in data_dict2.items()}
data_dict2["instance_2"] = data_dict2["instance_2"] + data_dict["instance"].max() # We want each instance to be individual, even between the two pcds
data_dict = {**data_dict, **data_dict2}

original_dict = deepcopy(data_dict)

# ----- Export both pcds to las -------
# pcd_to_las(data_dict["coord"], f"/workspaces/baseline/exp/temporary_export/1pcd_{data_dict['scene']}_{data_dict['frame']}.las", data_dict["segment"])
# pcd_to_las(data_dict["coord_2"], f"/workspaces/baseline/exp/temporary_export/2pcd_{data_dict['scene_2']}_{data_dict['frame_2']}.las", data_dict["segment_2"])

# ----- PolarMix Data Augmentation ----

# transform = PolarMixPaste(p=0, csv_stat_path="/workspaces/baseline/railseg/csv_stats/pedestrian_density_per_distance.csv")
# data_dict = transform(data_dict)
# pcd_to_las(data_dict["coord"], f"/workspaces/baseline/exp/temporary_export/pcd_after_{data_dict['scene']}_{data_dict['frame']}_plus_{data_dict['scene_2']}_{data_dict['frame_2']}.las", data_dict["segment"])

# ----- Sparsifying of track ----------
end_range=80
transform = Sparsify(end_range, track_label=4, p=1)
data_dict = transform(data_dict)
#pcd_to_las(data_dict["coord"], f"/workspaces/baseline/exp/temporary_export/pcd_after_sparse{end_range}_{data_dict['scene']}_{data_dict['frame']}_plus_{data_dict['scene_2']}_{data_dict['frame_2']}.las", data_dict["segment"])

# --- generate 2D plots for sprase track ---
print("Scene/frame:",data_dict["scene"], data_dict["frame"])

# Assuming original_dict and data_dict, color_map, and end_range are defined

# Extract data for the first plot
x1 = original_dict["coord"][:, 0]
y1 = original_dict["coord"][:, 1]
c1 = [np.array(color_map[segmentation]) / 255 for segmentation in original_dict["segment"]]

# Extract data for the second plot
x2 = data_dict["coord"][:, 0]
y2 = data_dict["coord"][:, 1]
c2 = [np.array(color_map[segmentation]) / 255 for segmentation in data_dict["segment"]]

# Create a figure with two subplots (one on top of the other)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [1, 1]})

# Plot the first set of data on the first axis
scatter1 = ax1.scatter(x1, y1, c=c1, s=1)
ax1.set_xlim(-10, 150)
ax1.set_ylim(-20, 20)
ax1.set_aspect('equal')
ax1.set_title("Original point cloud", fontsize=16)
ax1.set_xlabel('X [m]', fontsize=16)
ax1.set_ylabel('Y [m]', fontsize=16)

# Plot the second set of data on the second axis
scatter2 = ax2.scatter(x2, y2, c=c2, s=1)
ax2.set_xlim(-10, 150)
ax2.set_ylim(-20, 20)
ax2.set_aspect('equal')
ax2.set_title(f"Tracks sparsified with density level of range [{end_range-10}; {end_range}] m", fontsize=16)
ax2.set_xlabel('X [m]', fontsize=16)
ax2.set_ylabel('Y [m]', fontsize=16)

# Set tick size for both axes
ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)

# Create legend with circle markers
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=np.array([211, 211, 211]) / 255, markersize=10, label='Background'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=np.array([255, 140, 0]) / 255, markersize=10, label='Catenary pole'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=np.array([255, 0, 255]) / 255, markersize=10, label='Track')
]

# Add the legend to the first subplot in the upper left corner
ax2.legend(handles=legend_elements, loc='lower right', fontsize=14)

# Adjust layout to reduce space between plots and around the figure
fig.tight_layout()#(pad=1, h_pad=-1, w_pad=0.1)

# Save the figure
fig.savefig("/workspaces/baseline/exp/temporary_export/combined_2d_pcd.jpeg", dpi=400,bbox_inches="tight")
plt.close(fig)
