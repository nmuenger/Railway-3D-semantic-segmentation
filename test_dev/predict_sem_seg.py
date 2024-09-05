import os
import torch
import numpy as np
import torch
import raillabel
from pathlib import Path
from railseg import pcd_processing
import argparse
import yaml
import time
import fig_misc
from sklearn.metrics import confusion_matrix
from pointcept.utils.misc import intersection_and_union
import matplotlib.pyplot as plt
import pandas as pd
import shutil

from collections import OrderedDict
import pointcept.utils.comm as comm
from pointcept.datasets import transform
from pointcept.engines.defaults import (default_config_parser,default_setup)
from pointcept.models.builder import build_model
from pointcept.models.losses import build_criteria
from railseg.pcd_processing import normalize_multisensor_intensity

class extreme_IOU_and_loss_tracker:
    def __init__(self, cfg):
        self.metric_status = {"mIOU": 
                                {"highest":
                                    {"value" : 0, "scene":None},
                                 "lowest":
                                    {"value": 1, "scene": None}},
                              "loss":
                                {"highest":
                                    {"value": 0, "scene": None},
                                "lowest":
                                    {"value": 1000, "scene": None}}}
        self.all_intersection = np.zeros(cfg.num_classes)
        self.all_union = np.zeros(cfg.num_classes)

        self.criteria = build_criteria(cfg.model.criteria)

    def __call__(self, intersection, union, gt_segm, seg_logits, scene):

        loss = self.criteria(torch.Tensor(seg_logits.cpu()), torch.Tensor(gt_segm).type(torch.LongTensor))
        mIOU = np.mean(intersection/(union+1e-5))
        
        self.all_intersection = self.all_intersection+intersection
        self.all_union = self.all_union+union
        self.update_metric("mIOU", mIOU, scene)
        self.update_metric("loss", loss, scene)

    def update_metric(self, metric_type, value, scene):
        # metric_type = "mIOU" or "loss"
        if value > self.metric_status[metric_type]["highest"]["value"]:
            self.metric_status[metric_type]["highest"]["value"] = value
            self.metric_status[metric_type]["highest"]["scene"] = scene
        elif value < self.metric_status[metric_type]["lowest"]["value"]:
            self.metric_status[metric_type]["lowest"]["value"] = value
            self.metric_status[metric_type]["lowest"]["scene"] = scene
    
    def print_results(self):
        iou_class = self.all_intersection / (self.all_union + 1e-10)
        print(f'All ious per class: {iou_class}, \
                Best mIOU scene: {self.metric_status["mIOU"]["highest"]["scene"]}, with value: {self.metric_status["mIOU"]["highest"]["value"]} \
               \nWorst mIOU scene: {self.metric_status["mIOU"]["lowest"]["scene"]}, with value: {self.metric_status["mIOU"]["lowest"]["value"]}\
               \nWorst (highest) loss: {self.metric_status["loss"]["highest"]["scene"]}, with value: {self.metric_status["loss"]["highest"]["value"]}\
               \nBest (lowest) loss: {self.metric_status["loss"]["lowest"]["scene"]}, with value: {self.metric_status["loss"]["lowest"]["value"]}')
        
# For infering point cloud semantic segmentation
def inference(pcd_path, model, cfg, normalize_intensity=True, grid_size=0.05):
    pcd_path=Path(pcd_path)

    model.eval()
    
    # Load point cloud
    with open(pcd_path, "r") as b:
        scan = np.loadtxt(b, skiprows=11, usecols=(0,1,2,3,5)) # -> x|y|z|intensity|sensor_id
    coord = scan[:,:3]
    strength = scan[:,3:4]
    sensor_id = scan[:,4]
    input_dict = {"coord": coord,      #The coord of each point
                  "pts_idx": np.array(range(len(scan)))} 

    grid_sampling = transform.GridSample(keys=["coord","pts_idx"],return_grid_coord=True)    
    grid_sampled_dict = grid_sampling(input_dict)
    selected_pts_idx = grid_sampled_dict["pts_idx"]

    if normalize_intensity:
        strength = normalize_multisensor_intensity(strength, sensor_id)
    features = np.concatenate((coord,strength), axis=1)
    
    input_dict = {"feat":torch.tensor(features[selected_pts_idx], dtype=torch.float32), #The features of each points: x,y,z & intensity
                  "offset": torch.tensor([len(scan[selected_pts_idx])]),    #Read description of offset on README. For single sample, the length of the point cloud
                  "coord":torch.tensor(scan[selected_pts_idx,:3], dtype=torch.float32), #The coord of each point 
                  "grid_coord":torch.tensor(grid_sampled_dict["grid_coord"])}

    # Move point cloud to CUDA and run inference
    for key in input_dict.keys():
        if isinstance(input_dict[key], torch.Tensor):
            input_dict[key] = input_dict[key].cuda(non_blocking=True)
    with torch.no_grad():
        output_dict = model(input_dict)

    output = output_dict["seg_logits"]
    pred = output.max(1)[1]

    # Load Ground Truth
    scene_path = Path(pcd_path).parents[1] # Obtain .json file containing label for the current scene
    label_file = scene_path / (str(scene_path.name)+"_labels.json")

    segment = np.full(len(scan), 0) # Array containing the labels for each point -> first filled with the background tag
    current_instance = 0
    instances = np.full(len(scan), current_instance) # Assign a number for each individual object -> first all filled with 0
    segment_person_status = np.full(len(scan),"undefined")
    frame_nb = -1 # Undefined
    
    if os.path.exists(label_file):
        scene = raillabel.load(label_file) # Load json annotations for scene
        frame_nb = int(pcd_path.name.split('_')[0]) # returns the frame number as int '037'-> 37 
        scene_filtered = raillabel.filter(scene, include_frames=[frame_nb], include_annotation_types=['seg3d'])
        
        if frame_nb in scene_filtered.frames: # One of the point cloud doesnt have any frame annotation
            frame_objects = scene_filtered.frames[frame_nb].annotations.keys()
            
            for object in frame_objects:  
                label_name = scene_filtered.frames[frame_nb].annotations[object].object.type # Object type name ('track','train'...)
                learning_map = cfg["learning_map"]
                label_number = learning_map[label_name] # Map name to tag ('person' -> 1, ...)

                if label_number != cfg["annotation_disregarded"]:
                    current_instance += 1
                    pts_idx = scene_filtered.frames[frame_nb].annotations[object].point_ids # Points index for the object
                    segment[pts_idx] = label_number   # Switch corresponding points with the class number
                    instances[pts_idx] = current_instance 

                    if label_name=="person":
                        pose = scene_filtered.frames[frame_nb].annotations[object].attributes["pose"]
                        #pose_dict = {"upright":1, "sitting":2, "lying":3, "other":4}
                        segment_person_status[pts_idx] = pose

        else:
            pass # Keep all points of the point cloud as ignore_index
    else:
        pass # If the file doesn't exist, no label point exist, we keep the segment filled with ignore_index
    
    gt_segm = segment[selected_pts_idx]
    segment_person_status = segment_person_status[selected_pts_idx]
    coord = input_dict["coord"].cpu().numpy()
    pred_segm = pred.cpu().numpy()
    frame_name = scene_path.name+"_frame_"+str(frame_nb)

    # ignore_transform = transform.SparsifyIgnore(end_range=60)
    # data_dict_specific = {"coord": coord, "segment": gt_segm.copy(), "instance":instances[selected_pts_idx]}
    # data_dict_specific = ignore_transform(data_dict_specific)
    # criteria = build_criteria(cfg.model.criteria)
    # loss = criteria(torch.Tensor(output.cpu()), torch.Tensor(gt_segm).type(torch.LongTensor))
    # loss2 = criteria(torch.Tensor(output.cpu()), torch.Tensor(data_dict_specific["segment"]).type(torch.LongTensor))
    # print(f"loss 1: {loss}, loss2 {loss2}")
    # return coord, data_dict_specific["segment"], pred_segm, frame_name, segment_person_status

    return coord, gt_segm, pred_segm, frame_name, segment_person_status, output

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-cfg", default="test_dev/inference_configs/inference_cfg.yml", help="Path to yaml config file. Default: test_dev/inference_configs/nference_cfg.yml")
    args = parser.parse_args()
    yml_cfg_path = args.cfg

    with open(yml_cfg_path, 'r') as file:
        yml_cfg = yaml.safe_load(file)
    
    config_path =  yml_cfg["config_path"]
    weight_path = yml_cfg["weight_path"]
    out_path = yml_cfg["out_path"]

    Path(out_path).mkdir(parents=True,exist_ok=True)
    shutil.copyfile(yml_cfg_path, os.path.join(out_path,"inference_cfg_used.yml"))

    print(f"Using weights: {weight_path} and config:", {config_path})
    
    # Setup config
    cfg = default_config_parser(config_path,
                                options={"weight": weight_path})
    
    cfg = default_setup(cfg)

    # Build Model
    model = build_model(cfg.model).cuda()

    # Load Weights
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print((f"Num params: {n_parameters}"))
    if os.path.isfile(cfg.weight):
        print(f"Loading weight at: {cfg.weight}")
        checkpoint = torch.load(cfg.weight)
        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            if key.startswith("module."):
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
            else:
                if comm.get_world_size() > 1:
                    key = "module." + key  # xxx.xxx -> module.xxx.xxx
            weight[key] = value
        model.load_state_dict(weight, strict=True)
        print("=> Loaded weight '{}' (epoch {})".format(cfg.weight, checkpoint["epoch"]))
    else:
        raise RuntimeError("=> No checkpoint found at '{}'".format(cfg.weight))

    data_path = yml_cfg["data_path"]
    if yml_cfg["run_all_val_set"]:
        evaluated_frames = []
        for scene in yml_cfg["val_set"]:
            lidar_scene_path = os.path.join(data_path, scene, "lidar")
            evaluated_frames += [os.path.join(lidar_scene_path, frame) for frame in os.listdir(lidar_scene_path)]
    else:
        evaluated_frames = [os.path.join(data_path, frame) for frame in yml_cfg["evaluated_frames"]]
    
    num_classes = cfg.num_classes

    error_map = np.zeros((num_classes, 130,40)) 
    gt_map = np.zeros((num_classes, 130,40)) + 0.0001 #Avoid division by zero
    x_min, x_max = -10, 120
    y_min, y_max = -20, 20
    
    conf_mat = np.zeros((num_classes, num_classes))

    scene_conf_mat = np.zeros((num_classes, num_classes))
    prev_scene = evaluated_frames[0].rsplit("/")[-3]

    person_poses = ["upright", "sitting","lying", "other"]
    dict_intersection_pose = {k:0 for k in person_poses}
    dict_gt_pose = {k:0 for k in person_poses}

    all_intersection = []
    all_union = []
    all_target = []

    iou_tracker = extreme_IOU_and_loss_tracker(cfg)

    for pcd_path in evaluated_frames:
        current_scene = pcd_path.rsplit("/")[-3]

        if current_scene != prev_scene: # Save scene confusion matrix    
            if yml_cfg["save_confusion_matrix"]:
                fig_misc.export_confusion_mat(scene_conf_mat, cfg.learning_map_inv.values(), out_path, scene=prev_scene)
        
            prev_scene = current_scene
            scene_conf_mat = np.zeros((num_classes, num_classes))

        print("Running inference for point cloud:", pcd_path)

        coord, gt_segm, pred_segm, frame_name, segment_person_status, seg_logits = inference(pcd_path, model, cfg, normalize_intensity=yml_cfg["normalize_intensity"])

        intersection, union, target = intersection_and_union(pred_segm, gt_segm, K=num_classes)

        iou_tracker(intersection,union,gt_segm,seg_logits,pcd_path)

        for list, val in zip([all_intersection, all_union, all_target], [intersection, union, target]):
            list.append(val)

        for pose in person_poses:
            mask = np.where(segment_person_status == pose)
            dict_intersection_pose[pose] += sum(gt_segm[mask]==pred_segm[mask])
            dict_gt_pose[pose] += sum(gt_segm[mask])

        for k in range(num_classes):
            # TODO: Put this in a single function
            # In the error map, the (x_min, y_min)---(y=0)--->(y_max)
            #                        |                 |
            #                       (x=0)----------(x=0,y=0)
            #                        |   
            #                        v
            #                       (x_max)                     

            class_in_gt_mask = (gt_segm == k)
            misclassified_point_mask = (class_in_gt_mask & ~(gt_segm==pred_segm))
            # In 2D, map all the points to their lower 2D int coordinates.
            mapped_2d_coord_misc = np.floor(coord[misclassified_point_mask][:,[0,1]])
            mapped_2d_coord_gt = np.floor(coord[class_in_gt_mask][:,[0,1]])
            # Keep only point inside the frame
            misclassified_coord_in_border = mapped_2d_coord_misc[(mapped_2d_coord_misc[:,0]>=x_min) & (mapped_2d_coord_misc[:,0]<x_max) & (mapped_2d_coord_misc[:,1]>=y_min) & (mapped_2d_coord_misc[:,1]<y_max)]
            gt_coord_in_border = mapped_2d_coord_gt[(mapped_2d_coord_gt[:,0]>=x_min) & (mapped_2d_coord_gt[:,0]<x_max) & (mapped_2d_coord_gt[:,1]>=y_min) & (mapped_2d_coord_gt[:,1]<y_max)]
            # Then find number total number of error for each coordinates
            misclassified_coord, misclassified_count = np.unique( misclassified_coord_in_border , axis=0, return_counts=True)
            gt_coord, gt_count = np.unique( gt_coord_in_border , axis=0, return_counts=True)
            
            misclassified_coord = (misclassified_coord-[x_min, y_min]).astype(int) # Shift the coordinate so that it fit in the numpy array.
            gt_coord = (gt_coord-[x_min, y_min]).astype(int) # Shift the coordinate so that it fit in the numpy array.

            x_misc = misclassified_coord[:, 0]
            y_misc = misclassified_coord[:,1]
            x_gt = gt_coord[:,0]
            y_gt = gt_coord[:,1]

            error_map[k, x_misc, y_misc] += misclassified_count
            gt_map[k, x_gt, y_gt] += gt_count

        conf_mat = conf_mat + confusion_matrix(gt_segm,pred_segm, labels=np.arange(num_classes))
        scene_conf_mat = scene_conf_mat + confusion_matrix(gt_segm,pred_segm, labels=np.arange(num_classes))

        if yml_cfg["export_las_file"]: # Save predicted point cloud to LAS file
            Path(os.path.join(out_path, "predicted_pcd")).mkdir(parents=True, exist_ok=True) # makedir if doesn't exist
            exp_las_dir = os.path.join(out_path, "predicted_pcd", frame_name+".las")
            pcd_processing.pcd_to_las(coord, exp_las_dir, gt_segm, pred_segm)
            print("Predicted point cloud in LAS format saved under:", exp_las_dir,"\n")

        if yml_cfg["export_projected_pcd"]:
            fig_misc.project_2d_points(coord, pcd_path,out_path, frame_name, pred_segm, gt_segm, cfg)

    iou_tracker.print_results()

    if yml_cfg["save_pose_stat"]:
        fig_misc.export_person_pose_stat(dict_intersection_pose, dict_gt_pose, out_path)

    if yml_cfg["save_error_map"]:
        for k in range(num_classes):
            # To make the data understandable to heatmap function, the error map must be Transposed an the y axis flipped
            GT_points_map = np.flip(gt_map[k,:,:].T, axis=0)           # equivalent to TP+FN
            plot_error_map = np.flip(error_map[k,:,:].T, axis=0)       # equivalent FN
            plot_normalised_map = np.divide(plot_error_map, GT_points_map) # equivalent to FN/TP+FN-> FNR
            plot_normalised_map[GT_points_map<1]=np.NaN # In normalised map,set cell where no GT points are to Nan

            TP_points_map  = GT_points_map-plot_error_map
            plot_recall_map = TP_points_map/GT_points_map # equivalent to TP/TP+FN
            plot_recall_map[GT_points_map<1]=np.NaN # In Recall map,set cell where no GT points are to Nan

            fig_misc.export_error_map(plot_error_map, cfg.learning_map_inv[k], out_path, mode="error_map")
            fig_misc.export_error_map(plot_normalised_map, cfg.learning_map_inv[k], out_path, mode="FNR")
            fig_misc.export_error_map(plot_recall_map, cfg.learning_map_inv[k], out_path, mode="Recall_map")

            if k ==4:
                np.save(os.path.join(out_path,"GT_map_track.npy"), GT_points_map)
                np.save(os.path.join(out_path,"recall_map_track.npy"), plot_recall_map)


    if yml_cfg["save_gt_map"]:
        for k in range(num_classes):
            # To make the data understandable to heatmap function, the error map must be Transposed an the y axis flipped
            
            plot_gt_map = np.flip(gt_map[k,:,:].T, axis=0)
            # plot_gt_map = (plot_gt_map>=1) # Summarize the plot to presence or abesnce of point in the scene
            fig_misc.export_error_map(plot_gt_map, cfg.learning_map_inv[k], out_path, mode="debug_gt_map")


    if yml_cfg["save_confusion_matrix"]:
        fig_misc.export_confusion_mat(conf_mat, cfg.learning_map_inv.values(), out_path, scene=None)

    if yml_cfg["save_stat_metric"]: #Export iou, recall and precision to a csv file
        all_intersection, all_target, all_union = np.array(all_intersection).sum(axis=0), np.array(all_target).sum(axis=0), np.array(all_union).sum(axis=0)
        iou_class = all_intersection/(all_union + 1e-10)
        recall_class = all_intersection/(all_target + 1e-10)
        precision_class = all_intersection/(all_union-all_target+all_intersection+1e-10)
        
        df = pd.DataFrame([iou_class, recall_class, precision_class], columns=cfg.learning_map_inv.values())
        df.index = ["iou","recall","precision"]
        
        df.to_csv(os.path.join(out_path,"stats.csv"))

    

if __name__ == "__main__":
    
    time0 = time.time()
    main()
    print(f"Took {time.time()-time0:.2f} sec to run inference procedure.")

