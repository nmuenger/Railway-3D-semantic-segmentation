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
import wandb

def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()

def set_cfg_and_launch(args):
    if "sweep" in args.options:
        wandb.init(project="sweep")
        sweep_cfg = wandb.config
        # Overwrite here the parameters provided in the config file by using the exact same structure and the sweep value
        args.options["augmented_sampler"]=dict(active = True,
                                            augmentations_list = [dict(type=["PolarMixPaste"], augment_ratio=sweep_cfg.augmentation_1_ratio),                          
                                                #dict(type=["SparsifyTrackIgnore"], augment_ratio=0.2),
                                                #dict(type=["PolarMixPaste", "SparsifyTrackIgnore"], augment_ratio=0.4)
                                                ])
        default_lr = sweep_cfg.default_lr
        attention_lr = default_lr/10
        args.options["optimizer"] = dict(type="AdamW", lr=default_lr, weight_decay=0.005)
        args.options["scheduler"] = dict(
            type="OneCycleLR",
            max_lr=[default_lr, attention_lr],
            pct_start=0.04,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=100.0,
        )
        args.options["param_dicts"] = [dict(keyword="block", lr=attention_lr)]
        args.options["save_path"] = "exp/"+"sweep_lr_"+str(sweep_cfg.default_lr)+"_augment_ratio_"+str(sweep_cfg.augmentation_1_ratio)

    cfg = default_config_parser(args.config_file, args.options)

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
                                                "augmentation_1_ratio": {"values":[0.6,0.7,0.8,0.9,1]},
                                                "default_lr": {"values":[0.001]},
                                },
                            }
        
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="sweep")
        wandb.agent(sweep_id, function=lambda: set_cfg_and_launch(args))
    
    else:
        set_cfg_and_launch(args)

if __name__ == "__main__":
    main()
