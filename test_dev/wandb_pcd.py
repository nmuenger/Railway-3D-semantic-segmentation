# This is a dumb test to understand how 

import numpy as np
import os
import raillabel
import sys
sys.path.append('/workspaces/baseline/') 
print(sys.path)
from railseg import pcd_processing 

import wandb

learning_map = {
            'person':            1,
            'crowd' :            2,
            'train' :            3,
            'wagons':            4,
            'bicycle':           5,
            'group_of_bicycles': 6,
            'motorcycle':        7,
            'road_vehicle':      8,
            'animal':            9,
            'group_of_animals': 10,
            'wheelchair':       11,
            'drag_shoe':        12,
            'track':            13,
            'transition':       14,
            'switch':           15,
            'catenary_pole':    16,
            'signal_pole':      17,
            'signal':           18,
            'signal_bridge':    19,
            'buffer_stop':      20,
            'flame':            21,
            'smoke':            22
        }



wandb.login()


# We can only extract the x,y,z, intensity if we want.
pcd_path = "/workspaces/baseline/data/OSDaR_dataset/original/1_calibration_1.1/lidar/012_1631441453.299504000.pcd"
pcd_data = np.loadtxt(pcd_path, skiprows=11, usecols=(0,1,2,3,5)) # Skip the header rows



scene = raillabel.load('/workspaces/baseline/data/OSDaR_dataset/v_2/1_calibration_1.1/1_calibration_1.1_labels.json')
scenes_filtered_frame = raillabel.filter(scene, include_frames=[12], include_annotation_types=['seg3d'])
frame_objects = scenes_filtered_frame.frames[12].annotations.keys()
# Add one empty column filled with zeros for the classes
pcd_data = np.c_[pcd_data,np.zeros(len(pcd_data))]
#annotated_points_idx = ['background']*len(pcd_data)
points_annotations = np.full(len(pcd_data), 'background').astype('<U20') #np.zeros(len(pcd_data))

for object in frame_objects:

    pts_idx = scenes_filtered_frame.frames[12].annotations[object].point_ids
    label_name = scenes_filtered_frame.frames[12].annotations[object].object.type
    label_number = learning_map[label_name]
    #if label_number<=13:
    points_annotations[pts_idx] = label_name#label_number
    #annotated_points_idx[pts_idx] = label_number

wandb_pcd = pcd_processing.to_wandb_format(pcd_data, labels_list = points_annotations)


# wandb_pcd = pcd_data[:,[0,1,2,5]]
# wandb_pcd[:,3] += 1

#wandb_pcd[:,3] +=1 
  # 🐝 1️⃣ Start a new run to track this script
wandb.init(
    # Set the project where this run will be logged
    project="testing_point_cloud_logging", 
  
    # Track hyperparameters and run metadata
    config={
    "point_cloud_name": os.path.basename(pcd_path)
    })



wandb.log({"point_cloud": wandb.Object3D(wandb_pcd)})

# Mark the run as finished
wandb.finish()