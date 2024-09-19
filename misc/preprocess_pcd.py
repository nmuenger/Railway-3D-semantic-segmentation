"""
This file has for purpose to preprocess the data contained in separate files, i.e. lidar point cloud (.pcd) and annotation (.json)
into a single one, in binary format. This is done to make the process of loading data during training faster.
For a point cloud of N points, the resulting array will be of shape [N, 7]
(X | Y | Z | intensity | classification | instance | person pose | sensor_id)
"""

import numpy as np
import raillabel
import time
import os

DATA_DIR = "data/OSDaR23_dataset"
EXPORT_DIR = "/workspaces/baseline/exp/preprocessed_pcd"

# --- Definition of corresponding label number -----
ANNOTATION_DISREGARDED = -2
learning_map = {
            'background':         0,
            'person':             1,
            'crowd':              1,
            'train':              2,
            'wagons':             2,
            'bicycle':            0, # Very low number of annotations
            'group_of_bicycles':  0, # No annotations
            'motorcycle':         0, # No annotations
            'road_vehicle':       3, 
            'animal':             0, # Very low number of annotations
            'group_of_animals':   0, # No annotation
            'wheelchair':         0, # No annotation
            'drag_shoe':          0, # No annotation
            'track':              4, 
            'transition':         4, 
            'switch':             ANNOTATION_DISREGARDED, # Causes inconsistent annotation otherwise
            'catenary_pole':      5, 
            'signal_pole':        6, 
            'signal':             6,
            'signal_bridge':      0, # Very low number of annotations
            'buffer_stop':        7, 
            'flame':              0, # No annotations
            'smoke':              0  # No annotations
        }

pose_dict = {"upright":0, "sitting":1, "lying":2, "other":3} # Person position in the dataset

files_to_skip = [os.path.join(DATA_DIR, pcd) 
                 for pcd in ["15_construction_vehicle_15.1/lidar/075_1631531288.599807000.pcd"]] # Those files have an issue and should not be taken into account


def load_annotations(lidar_file_path, json_file_path):
    """Given the lidar path and the annotation file path, return the numpy array with the desired info, i.e.:
        X | Y | Z | intensity | classification | instance | person pose | sensor_id"""
    
    with open(lidar_file_path, "r") as b:
        scan = np.loadtxt(b, skiprows=11, usecols=(0,1,2,3,5)) # -> x|y|z|intensity|sensor_id
    
    coord = scan[:, :3]     # The x,y,z coordinates of the points 
    strength = scan[:, 3].reshape([-1, 1]) # The intensity of the points 
    sensor_id = scan[:, 4].reshape([-1, 1]) # The id indicating which lidar sensor captured the point

    segment = np.full([len(scan), 1], 0) # Array containing the labels for each point ->first filled with the background tag
    current_instance = 0
    instances = np.full([len(scan), 1], current_instance) # Assign a number for each individual object -> first all filled with 0
    segment_person_status = np.full([len(scan), 1], -1) # Only adding pose attribute if the point is of type "person", else: -1
    
    scene = raillabel.load(json_file_path) # Load json annotations for scene
    frame_nb = int(lidar_file_path.split("/")[-1].split("_")[0]) # returns the frame number as int '037'-> 37 
    scene_filtered = raillabel.filter(scene, include_frames=[frame_nb], include_annotation_types=['seg3d'])
    
    if frame_nb in scene_filtered.frames: # One of the point cloud doesnt have any frame annotation
        frame_objects = scene_filtered.frames[frame_nb].annotations.keys()
        
        for object in frame_objects:  
            label_name = scene_filtered.frames[frame_nb].annotations[object].object.type # Object type name ('track','train'...)
            label_number = learning_map[label_name] # Map name to tag ('person' -> 1, ...)

            if label_number != ANNOTATION_DISREGARDED: # Only change point attribute if it is desired
                current_instance += 1
                pts_idx = scene_filtered.frames[frame_nb].annotations[object].point_ids # Points index for the object
                segment[pts_idx] = label_number   # Switch corresponding points with the class number
                instances[pts_idx] = current_instance 

                if label_name=="person":
                    pose = scene_filtered.frames[frame_nb].annotations[object].attributes["pose"]
                    segment_person_status[pts_idx] = pose_dict[pose]

    else:
        pass # Keep all points of the point cloud as ignore_index
    
    annotated_pcd_info = np.hstack((coord, strength, segment, instances, segment_person_status, sensor_id))

    return(annotated_pcd_info)

    
if __name__ == "__main__":
    print("Starting the preprocessing of the data...")
    time_0 = time.time()

    for scene in os.listdir(DATA_DIR):
        print("Processing scene:", scene)
        scene_input_dir = os.path.join(DATA_DIR, scene)
        scene_export_dir = os.path.join(EXPORT_DIR,scene)

        if not os.path.exists(scene_export_dir):
            os.makedirs(scene_export_dir)   
        
        json_file = os.path.join(scene_input_dir, scene_input_dir.rsplit("/", maxsplit=1)[1]+"_labels.json")

        if os.path.exists(json_file):
            for lidar_frame in os.listdir(os.path.join(scene_input_dir, "lidar")):
                lidar_frame_dir = os.path.join(scene_input_dir, "lidar", lidar_frame)
                if lidar_frame_dir not in files_to_skip:
                    preprocessed_pcd = load_annotations(lidar_frame_dir, json_file)
                else:
                    print("Skipped frame:", lidar_frame_dir)
                    continue

                preprocessed_pcd.tofile(os.path.join(scene_export_dir, lidar_frame+".bin")) # Exporting

    print(f"Finished preprocessing all the scenes in {round(time.time()-time_0,2)} sec.")
