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
from pointcept.datasets.transform import PolarMix, Sparsify

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

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
class Sparsify(object): # Method inspire by "Part-Aware Data Augmentation for 3D Object Detection in Point Cloud"
    def __init__(self, end_range=40, track_label=4):
        self.end_range = end_range 
        self.track_label = track_label
    
    def __call__(self, data_dict):
        """Goal of this function is to sparsify the track annotations closer to the LiDAR sensor. 
        It evaluates number of points between [end_range-10 ;end_range] and applies the same density for each bucket of 10m before this."""
        
        track_points = np.where(data_dict["segment"]==self.track_label)[0] # Find index of points of class "track"
        track_instances = np.unique(data_dict["instance"][track_points]) # Retrieve the instance number of each track element

        x_y_curve = []
        for inst in track_instances: 
            instance_pt_idx = np.where(data_dict["instance"]==inst)[0]
            x = data_dict["coord"][instance_pt_idx,0]
            y = data_dict["coord"][instance_pt_idx,1]

            # Fit a polynomial regression model
            degree = 2  # You can change the degree based on your needs
            poly_features = PolynomialFeatures(degree=degree)
            x_poly = poly_features.fit_transform(x.reshape(-1, 1))
            poly_model = LinearRegression()
            poly_model.fit(x_poly, y)

            # Predict y values for a smooth curve
            x_smooth = np.linspace(x.max(), x.max()+10, 300)
            x_smooth_poly = poly_features.transform(x_smooth.reshape(-1, 1))
            y_smooth = poly_model.predict(x_smooth_poly)
            x_y_curve.append({"x":x_smooth_poly, "y":y_smooth })
       
    
        return x_y_curve
        

# ----------- Start of the code ------------
random.seed(32)

data = OSDaR23Dataset(data_root="/workspaces/baseline/exp/preprocessed_pcd", learning_map=learning_map)
rand_int1 = random.randint(0, 1048)

rand_int1=240
data_dict = data.get_data(rand_int1) # was 140 before


# ----- Export both pcds to las -------
# pcd_to_las(data_dict["coord"], f"/workspaces/baseline/exp/temporary_export/pcd_{data_dict['scene']}_{data_dict['frame']}.las", data_dict["segment"])

# ----- Sparsifying of track ----------
end_range=80
transform = Sparsify(end_range, track_label=4)
x_y_curve = transform(data_dict)
#pcd_to_las(data_dict["coord"], f"/workspaces/baseline/exp/temporary_export/pcd_after_sparse{end_range}_{data_dict['scene']}_{data_dict['frame']}_plus_{data_dict['scene_2']}_{data_dict['frame_2']}.las", data_dict["segment"])

# --- generate 2D plots for sprase track ---
x = data_dict["coord"][:,0]
y = data_dict["coord"][:,1]
c = [np.array(color_map[segmentation])/255 for segmentation in data_dict["segment"]]
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)
scatter = ax.scatter(x, y, c=c, s=0.1)
ax.set_xlim(-10, 150)
ax.set_ylim(-20, 20)
ax.set_aspect('equal')
for i in range(len(x_y_curve)):
    ax.plot(x_y_curve[i]["x"], x_y_curve[i]["y"], color='blue', label='Polynomial Fit')
fig.suptitle("Original point cloud")
plt.xlabel('X [m]', fontsize=12)
plt.ylabel('Y [m]', fontsize=12)
fig.tight_layout()
fig.savefig("/workspaces/baseline/exp/temporary_export/predict_track.jpeg", dpi=300)
plt.close(fig)

# x = new_data_dict["coord"][:,0]
# y = new_data_dict["coord"][:,1]
# c = [np.array(color_map[segmentation])/255 for segmentation in new_data_dict["segment"]]
# fig = plt.figure(figsize=(10, 4))
# ax = fig.add_subplot(111)
# scatter = ax.scatter(x, y, c=c, s=0.05)
# ax.set_xlim(-10, 150)
# ax.set_ylim(-20, 20)
# ax.set_aspect('equal')
# fig.suptitle(f"Track overall sparsified to same level as between [{start_range};{start_range+10}]m")
# plt.xlabel('X [m]', fontsize=12)
# plt.ylabel('Y [m]', fontsize=12)
# fig.tight_layout()
# fig.savefig(f"/workspaces/baseline/exp/temporary_export/2d_pcd_after_{start_range}.jpeg", dpi=500)
# plt.close(fig)