"""
Main Testing Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.test import TESTERS
from pointcept.engines.launch import launch
import os
import wandb


def main_worker(cfg):
    cfg = default_setup(cfg)
    tester = TESTERS.build(dict(type=cfg.test.type, cfg=cfg))
    tester.test()



def set_cfg_and_launch(args):
    if "sweep" in args.options:
        augmentation_desc = "online_sparse"
        overall_folder_name = "/workspaces/baseline/exp/weights_sweeps_training/ONline_sparse_exp_0" # The path to the folder containing subfolder with weights in them
        
        
        
        wandb.init(project=augmentation_desc)
        sweep_cfg = wandb.config
        augm_value_str = "{:.1f}".format(sweep_cfg.augmentation_value)
        subfolder_name = "online_sparse_"+ augm_value_str # all the subfolders should follow the same name structure, e.g. if subfolder_name = "online_sparse_"-> online_sparse_0.0; online_sparse_0.1...
        weight_dir = os.path.join(overall_folder_name, subfolder_name,"model","model_last.pth")
        
        args.options["save_path"] = "exp/test_inferences/"+augmentation_desc + "_" + augm_value_str
        args.options["weight"] = weight_dir

    cfg = default_config_parser(args.config_file, args.options)
    
    # if "sweep" in args.options:
    #     weight_dir = os.path.join(overall_folder_name, subfolder_name,"model","model_last.pth")

    #     for i in range(len(cfg.data.train.transform)):
    #         if cfg.data.train.transform[i]["type"] == "PolarMixPaste":
    #             cfg.data.train.transform[i]["p"]=sweep_cfg.prob_augmentation_paste # Update probability of applying augmentation on the fly
    #             print("Modified paste transform with prob:", cfg.data.train.transform[i]["p"])

    #         if cfg.data.train.transform[i]["type"] == "Sparsify":
    #             cfg.data.train.transform[i]["p"]=sweep_cfg.prob_augmentation_sparse # Update probability of applying augmentation on the fly
    #             print("Modified sparse transform with prob:", cfg.data.train.transform[i]["p"])

    #cfg.dump(os.path.join(cfg.save_path, "config.py")) #Re-dump the config, with the updated values

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )   




def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    if "sweep" in args.options:
        sweep_configuration = {
                                "method": "grid",
                                "parameters": {
                                                "augmentation_value": {"values":[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]},
                                },
                            }
        
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="precise_test_sweep")
        wandb.agent(sweep_id, function=lambda: set_cfg_and_launch(args))
    
    else:
        set_cfg_and_launch(args)


if __name__ == "__main__":
    main()
