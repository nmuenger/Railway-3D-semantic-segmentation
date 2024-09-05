"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch
import os
import wandb

def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()

def set_cfg_and_launch(args):
    if "sweep" in args.options:
        wandb.init(project="new_sweep_set_sparse_restrained_50")
        sweep_cfg = wandb.config
       
        args.options["save_path"] = "exp/"+"new_sweep_online_augment_without_rotation_but_WITH_put_to_back_paste_"+str(sweep_cfg.prob_augmentation_paste)+"_sparse_"+str(sweep_cfg.prob_augmentation_sparse)
        

    cfg = default_config_parser(args.config_file, args.options)
    
    if "sweep" in args.options:
        for i in range(len(cfg.data.train.transform)):
            if cfg.data.train.transform[i]["type"] == "PolarMixPaste":
                cfg.data.train.transform[i]["p"]=sweep_cfg.prob_augmentation_paste # Update probability of applying augmentation on the fly
                print("Modified paste transform with prob:", cfg.data.train.transform[i]["p"])

            if cfg.data.train.transform[i]["type"] == "Sparsify":
                cfg.data.train.transform[i]["p"]=sweep_cfg.prob_augmentation_sparse # Update probability of applying augmentation on the fly
                print("Modified sparse transform with prob:", cfg.data.train.transform[i]["p"])

    cfg.dump(os.path.join(cfg.save_path, "config.py")) #Re-dump the config, with the updated values

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
    
    # To activate the sweep mode add --options sweep=True to your launch command
    # Define below the parameters that must be tested during the sweep. The must then be properly updated in the
    # config in the function set_cfg_and_launch()
    if "sweep" in args.options:
        sweep_configuration = {
                                "method": "grid",
                                "metric": {"goal": "minimize", "name": "val/total/mIoU"},
                                "parameters": {
                                                "prob_augmentation_paste": {"values":[1.0, 0.9, 0.8, 0.7]},
                                                "prob_augmentation_sparse": {"values":[0]}
                                },
                            }
        
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="sweep")
        wandb.agent(sweep_id, function=lambda: set_cfg_and_launch(args))
    
    else:
        set_cfg_and_launch(args)

if __name__ == "__main__":
    main()
