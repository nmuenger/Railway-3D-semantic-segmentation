import os
import torch
import numpy as np
import torch
from pathlib import Path
import pcd_processing
import argparse
import yaml
import time
import fig_misc
import shutil

from collections import OrderedDict
import pointcept.utils.comm as comm
from pointcept.datasets import transform
from pointcept.engines.defaults import (default_config_parser,default_setup)
from pointcept.models.builder import build_model

from pointcept.datasets import build_dataset


# For infering point cloud semantic segmentation
def inference(pcd_path, model, osdar_dataset=None ):
    pcd_path = Path(pcd_path)

    model.eval()
    
    if osdar_dataset is not None:
        input_dict = osdar_dataset.get_data_unprocessed(pcd_dir=pcd_path)
        scene_path = pcd_path.parents[1]
        frame_name = scene_path.name+"_frame_"+pcd_path.name.split("_")[0]
    else:
        input_dict={}
        with open(pcd_path, "r") as b:
            scan = np.loadtxt(b, skiprows=11, usecols=(0,1,2,3,4)) # -> x|y|z|intensity|tag
        tag_field = scan[:,4]
        mask=np.where(tag_field>0)[0]
        input_dict["coord"] = scan[mask, :3]     # The x,y,z coordinates of the points 
        input_dict["strength"] = scan[mask, 3].reshape([-1, 1])/255 # The intensity of the points (normalised)
        input_dict["segment"] = np.zeros(len(input_dict["strength"])).astype(int) # Modify to correct mask if provided
        input_dict["instance"] = np.zeros(len(input_dict["strength"])).astype(int) # Modify to correct mask if provided
        frame_name = pcd_path.name

    input_dict["feat"] = np.concatenate((input_dict["coord"],input_dict["strength"]), axis=1)
    # 'keys' are the field which should be transformed by the grid sampling
    grid_sampling = transform.GridSample(keys=["coord", "strength", "segment", "instance", "feat"],return_grid_coord=True)    
    input_dict  = grid_sampling(input_dict) # Update input_dict with grid-sampled point cloud
 
    to_tensor = transform.ToTensor()
    input_dict = to_tensor(input_dict)

    assert isinstance(input_dict, dict), "Expected input_dict to be a dictionary, but got: {}".format(type(input_dict).__name__)

    input_dict["offset"] = torch.tensor([len(input_dict["coord"])])

    # Move point cloud to CUDA and run inference
    for key in input_dict.keys():
        if isinstance(input_dict[key], torch.Tensor):
            input_dict[key] = input_dict[key].cuda(non_blocking=True)
    with torch.no_grad():
        output_dict = model(input_dict)

    output = output_dict["seg_logits"]
    pred = output.max(1)[1]

    gt_segm = input_dict["segment"].cpu().numpy()
    coord = input_dict["coord"].cpu().numpy()
    pred_segm = pred.cpu().numpy()
    
    return coord, gt_segm, pred_segm, frame_name

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-cfg", default="misc/inference_configs/inference_cfg.yml", help="Path to yaml config file. Default: misc/inference_configs/nference_cfg.yml")
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
    if yml_cfg["OSDaR_data"]: # If attribute == True, will load annoations etc. from osdar
        dataset = build_dataset(cfg.data.train)
    else:
        dataset = None
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
    
    for pcd_path in evaluated_frames:

        print("Running inference for point cloud:", pcd_path)

        coord, gt_segm, pred_segm, frame_name = inference(pcd_path, model, dataset)

        if yml_cfg["export_las_file"]: # Save predicted point cloud to LAS file
            Path(os.path.join(out_path, "predicted_pcd")).mkdir(parents=True, exist_ok=True) # makedir if doesn't exist
            exp_las_dir = os.path.join(out_path, "predicted_pcd", frame_name+".las")
            pcd_processing.pcd_to_las(coord, exp_las_dir, gt_segm, pred_segm)
            print("Predicted point cloud in LAS format saved under:", exp_las_dir,"\n")

        if yml_cfg["export_projected_pcd"]:
            fig_misc.project_2d_points(coord, pcd_path,out_path, frame_name, pred_segm, gt_segm, cfg)



if __name__ == "__main__":
    
    time0 = time.time()
    main()
    print(f"Took {time.time()-time0:.2f} sec to run inference procedure.")

