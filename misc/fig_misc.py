import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import numpy as np
import pandas as pd
import cv2
from scipy.spatial.transform import Rotation as R
import os
from pathlib import Path

def project_2d_points(pcd_coord, pcd_path, out_path,  frame_name, pred_segm=None, gt_segm=None, cfg=None):
    """pcd_coord = [Nx3]"""
    if cfg is not None:
        color_map = cfg.color_map
        learning_map_inv = cfg.learning_map_inv
    else: # Use default color and learning map
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


    homogeneous_pcd_coord = np.r_[pcd_coord.T, np.ones((1,len(pcd_coord)))]
    # Code modified from : https://github.com/DSD-DBS/raillabel/issues/26
    sensors = {
        "rgb_highres_center" : {
            "intrinsic" : np.array([7267.95450880415, 0.0, 2056.049238502414, 0.0, 0.0, 7267.95450880415, 1232.862908875167, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape(3, 4),
            "quaternion" : np.array([-0.00313306, 0.0562995, 0.00482918, 0.998397]),
            "translation" : np.array([0.0801578, -0.332862, 3.50982]).reshape(3, 1),
            "distortion_coeffs" : np.array([-0.0764891, -0.188057, 0.0, 0.0, 1.78279]),
            "width_px" : 4112,
            "height_px" : 2504
        },
    }

    details = sensors["rgb_highres_center"]

    intrinsic = details["intrinsic"]
    quaternion = details["quaternion"]
    translation = details["translation"]
    distortion_coeffs = details["distortion_coeffs"]
    width = details["width_px"]
    height = details["height_px"]

    # might be an unnecessary step
    intrinsics_undistorted, _ = cv2.getOptimalNewCameraMatrix(intrinsic[:,:-1], distortion_coeffs, (width,height), 1, (width,height))
    intrinsics_undistorted = np.hstack((intrinsics_undistorted, np.asarray([[0.0], [0.0], [0.0]])))

    rotation = R.from_quat(quaternion)
    coord_system_rot = R.from_euler('zxy', [-90, 0, 90], degrees=True)
    rotation = (rotation * coord_system_rot).as_matrix()

    extrinsic = np.vstack((np.hstack((rotation.T, - rotation.T @ translation)), [0.0,0.0,0.0,1.0]))
    projection = np.matmul(intrinsics_undistorted, extrinsic)

    
    points_2d = np.matmul(projection, homogeneous_pcd_coord)

    # divide x by z if z is not 0
    points_2d[0,:] = np.divide(points_2d[0,:],points_2d[2,:],where=points_2d[2,:]!=0)
    # divide y by z if z is not 0
    points_2d[1,:] = np.divide(points_2d[1,:],points_2d[2,:],where=points_2d[2,:]!=0)

    scene_path, _, lidar_file_name = pcd_path.rsplit("/",2)
    frame_nb = lidar_file_name.split("_")[0] 
    img_file = [filename for filename in os.listdir(os.path.join(scene_path,"rgb_highres_center")) if filename.startswith(frame_nb)]
    if len(img_file)>1:
        print("Error, found more than one matching image")
        exit()
    else:
        image_path = os.path.join(scene_path,"rgb_highres_center", img_file[0])
    
    frame_image = cv2.imread(image_path)
    
    # Prediction
    if pred_segm is not None: # Only create if prediction were provided
        fig, ax = plt.subplots()
        plt.imshow(cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB))
        ax.set_xlim([0, width])
        ax.set_ylim([0, height])
        present_classes = np.unique(pred_segm)
        for k in present_classes:
            mask=np.where(pred_segm==k)
            ax.scatter(points_2d[0,mask],points_2d[1,mask], c=np.array([color_map[k]])/255, label=learning_map_inv[k],s=0.1)

        Path(os.path.join(out_path,"figures/projected_pcd/")).mkdir(parents=True, exist_ok=True) # makedir if doesn't exist
        lgnd = plt.legend(scatterpoints=1, fontsize=12)
        for handle in lgnd.legendHandles:
            handle.set_sizes([40.0])
        plt.gca().invert_yaxis()
        plt.title(f"Prediction for frame: {frame_name}")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_path,f"figures/projected_pcd/{frame_name}_pred.jpeg"),dpi=500)
        plt.cla()

    # Groud Truth
    if gt_segm is not None: # Only create if GT mask is given
        fig, ax = plt.subplots()
        plt.imshow(cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB))
        ax.set_xlim([0, width])
        ax.set_ylim([0, height])
        present_classes = np.unique(gt_segm)
        for k in present_classes:
            mask=np.where(gt_segm==k)
            ax.scatter(points_2d[0,mask],points_2d[1,mask], c=np.array([color_map[k]])/255, label=learning_map_inv[k],s=0.1)

        lgnd = plt.legend(scatterpoints=1, fontsize=12)
        for handle in lgnd.legendHandles:
            handle.set_sizes([40.0])
        plt.gca().invert_yaxis()
        plt.title(f"Ground Truth for frame: {frame_name}")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_path,f"figures/projected_pcd/{frame_name}_gt.jpeg"),dpi=500)
        plt.close("all")



        