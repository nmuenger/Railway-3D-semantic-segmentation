import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import numpy as np
import pandas as pd
import cv2
from scipy.spatial.transform import Rotation as R
import os
from pathlib import Path

def export_error_map(error_map, class_name, export_folder, mode="error_map"):
    Path(os.path.join(export_folder, f"figures/error_map/")).mkdir(parents=True, exist_ok=True) # makedir if doesn't exist
    
    plt.figure(figsize=(17,5))
    if mode=="debug_gt_map":
        title = f"Location where a point ground truth class '{class_name}' is present, per grid cell of 1x1 m (Computed on entire validation set)"
        color_map_title="Debuging"
        export_path = os.path.join(export_folder, f"figures/error_map/gt_class_{class_name}.png")
    elif mode == "error_map":
        title = f"Number of misclassified points from ground truth class '{class_name}' per grid cell of 1x1 m (Computed on entire validation set)"
        color_map_title="# of False Negative points"
        export_path = os.path.join(export_folder, f"figures/error_map/class_{class_name}.png")
    elif mode== "FNR":
        title = f"Number of misclassified points from GT class '{class_name}' \ndivided by number of GT points for that class, per grid cell of 1x1 m \n(Computed on entire validation set)"
        color_map_title="False negative rate"
        export_path = os.path.join(export_folder, f"figures/error_map/class_{class_name}_normalised.png")
    elif mode=="Recall_map":
        title = f"Number of rightfully classifiedpoints from GT class '{class_name}' \ndivided by number of GT points for that class, per grid cell of 1x1 m \n(Computed on entire validation set)"
        color_map_title="Recall"
        export_path = os.path.join(export_folder, f"figures/error_map/class_{class_name}_recall.png")

    # This will do the "alt" plot version i
    ax = sns.heatmap(error_map, linewidth=0.,square=True, vmin=0,vmax=1, cmap="viridis_r")#,norm=LogNorm()) # Transpose so that x values are represented horizontaly, on the x axis. Uncomment last arugment to put in log norm
    #The two line below will do the original version style which was in the powerpoint for pres. 2
    # error_map[error_map==np.NaN] = 0
    # ax = sns.heatmap(error_map, linewidth=0.,square=True, vmax=1)
    
    ax.figure.axes[-1].set_ylabel(color_map_title, size=16)


    # ax.axhline(y = 0, color='k',linewidth = 1)  # add frame around heatmap
    # ax.axhline(y = 39.99, color = 'k', linewidth = 1) 
    # ax.axvline(x = 0, color = 'k',linewidth = 1) 
    # ax.axvline(x = 129.99,  color = 'k', linewidth = 1)  

    #ax.set_title(title, fontdict = {'fontsize': 16}, pad=60) #.set_title('Title', pad=20)
    cax = ax.figure.axes[-1] # Change size of ticks in colorbar for heatmap
    cax.tick_params(labelsize=14)
    plt.xlabel("X [m]",fontsize=16)
    plt.ylabel("Y [m]",fontsize=16)
    plt.xticks(np.arange(0,131,10), np.arange(-10,121,10),fontsize=14)
    plt.yticks(np.arange(0,41,10), np.arange(20,-21,-10),fontsize=14) # The y axis is inverted in the heatmp xompared to what we expect
    plt.savefig(export_path,bbox_inches='tight')
    plt.clf() # Clear figure for next creation
    plt.close('all')

def export_confusion_mat(conf_mat, labels, export_folder, scene = None):
    Path(os.path.join(export_folder,"figures/")).mkdir(parents=True, exist_ok=True) # makedir if doesn't exist
    if scene is None:
        title = "Confusion matrix of points classification"
        export_path = os.path.join(export_folder,f"figures/confusion_matrix.png")
    else: 
        Path(os.path.join(export_folder,"figures/scene_confusion_matrix")).mkdir(parents=True, exist_ok=True) # makedir if doesn't exist
        title = f"Confusion matrix of points classification for scene: {scene}"
        export_path = os.path.join(export_folder,f"figures/scene_confusion_matrix/cm_{scene}.png")
        
    df_conf_mat = pd.DataFrame(conf_mat, index=labels, columns=labels)
    plt.figure(figsize = (10,7))
    sns.heatmap(df_conf_mat, annot=True,norm=LogNorm())
    plt.title(title)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('GT', fontsize=12)
    plt.savefig(export_path, bbox_inches='tight')
    plt.close('all')

def export_person_pose_stat(dict_intersection_pose, dict_gt_pose, export_folder):

    accuracy_person = list(dict_intersection_pose.values())/(np.array(list(dict_gt_pose.values()))+1e-6) #*100 Uncomment last if you want values in percent
    
    Path(os.path.join(export_folder,"figures/")).mkdir(parents=True, exist_ok=True) # makedir if doesn't exist

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(dict_gt_pose.keys(),accuracy_person,)
    #plt.title("Percentage of accurately predicted 'person' point on val set")

    for i in range(len(dict_gt_pose.keys())):
        if list(dict_gt_pose.values())[i]==0:
            text = "N.A."
        else:
            text = f"{accuracy_person[i]:.2f}" #str(round(accuracy_person[i],2))

        ax.text(i, accuracy_person[i], text, ha = 'center',va="bottom",fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylim([0,1.0])
    ax.set_xlabel("Person's position",fontsize=16)
    ax.set_ylabel("Recall for class $\it{person}$",fontsize=16)
    
    fig.savefig(os.path.join(export_folder,"figures/person_poses_stats.png"), dpi=400, bbox_inches='tight')

    plt.close(fig)

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
    
    #TODO Put plot creation in single function
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



        