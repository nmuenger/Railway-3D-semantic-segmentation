"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import weakref
import torch
import torch.nn as nn
import torch.utils.data
import wandb
from datetime import datetime
from functools import partial
import torch.distributed as dist
from pointcept.utils.misc import intersection_and_union_gpu
import numpy as np

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage
from pointcept.utils.registry import Registry
from pointcept.datasets.osdar23 import AugmentedSampler

TRAINERS = Registry("trainers")


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        # This is the default trainer, not the one actually used when we launch the script...
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self, eval_only=False):
        for h in self.hooks:
            if type(h).__name__=="CheckpointSaver" and eval_only:
                continue
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            #self.writer.close()
            self.writer.finish()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)
        self.curr_iter = 0

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                if self.cfg.eval_only == False:
                    # => before epoch
                    # TODO: optimize to iteration based
                    if (comm.get_world_size() > 1) or (type(self.train_loader.sampler).__name__ == "AugmentedSampler"):
                        self.train_loader.sampler.set_epoch(self.epoch)
                    self.model.train()
                    self.data_iterator = enumerate(self.train_loader)
                    self.before_epoch()
                    # => run_epoch
                    for (
                        self.comm_info["iter"],
                        self.comm_info["input_dict"],
                    ) in self.data_iterator:
                        # => before_step
                        self.before_step()
                        # => run_step
                        self.run_step()
                        # => after_step
                        self.after_step()
                # => after epoch
                self.after_epoch(eval_only=self.cfg.eval_only)
            # => after train
            self.after_train()

    def run_step(self):
        self.curr_iter += 1

        input_dict = self.comm_info["input_dict"]
    
        if self.writer is not None:
            self.writer.log({"iter": self.curr_iter, "feature length": len(input_dict['feat'])})

        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"]
            
            # #----Added to upload metric also during training:-----
            output = output_dict["seg_logits"]
            del output_dict["seg_logits"] # The seg logits was added in output_dict to allow for computation of training mIoU, but needs to be removed for further step.
            pred = output.max(1)[1]
            segment = input_dict["segment"]

            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.cfg.data.num_classes,
                self.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Add a sort of registery for the class which are present in the ground truth
            class_presence = np.zeros_like(intersection)
            class_presence[segment.cpu().numpy()] = True # For all classes which are in ground truth, set to true.

            # Here there is no need to sync since sync happened in dist.all_reduce
            self.storage.put_scalar("train_intersection", intersection)
            self.storage.put_scalar("train_union", union)
            self.storage.put_scalar("train_target", target)
            self.storage.put_scalar("train_class_presence", class_presence) 
            # # -----End of code I added------
    
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict
        
    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info(f"Num params: {n_parameters}")
        # Create a DistributedDataParallel model if there are >1 processes.
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    # Original writer method:
    # def build_writer(self):
    #     writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
    #     self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
    #     return writer

    # Updated for WandB writer
    def build_writer(self):
        if self.cfg.sweep:
            self.logger.info("W&B has already been init due to sweep launch.")
            wandb.config.update(self.cfg) # Still want to have all the config params, not only sweep's
            return wandb
        
        if comm.is_main_process():
            if self.cfg.debug:
                mode = "disabled"
            else:
                mode = "online"
            experiment_name = datetime.now().strftime("%d/%m/%Y_%H:%M") # Set time as experiment name
            wandb.init(project=self.cfg.save_path.replace("/","_"), name=experiment_name, mode=mode)
            wandb.config.update(self.cfg)  # Log your configuration
        else:
            wandb.init(reinit=True, mode="disabled")  # Disable W&B for non-main processes
        self.logger.info(f"W&B logger initialized for experiment.")
        return wandb


    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            self.logger.info(f"Using distributed sampler for training")
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        elif self.cfg.augmented_sampler["active"]:
            self.logger.info(f"Using AugmentedSampler. The number of samples seen during one epoch will be larger than the training set size.")
            train_sampler = AugmentedSampler(train_data, self.cfg)
        else:
            self.logger.info("Using the regular pytorch sampler")
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=True,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        if type(self.train_loader.sampler).__name__!="AugmentedSampler": 
            self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        else:
            if self.cfg.epoch!=self.cfg.eval_epoch:
                sys.exit(f"Error: The total number of epoch is not equal to number of eval epoch. This behaviour is currently not supported for the AugmentedSampler...")
            self.cfg.scheduler.total_steps = self.train_loader.sampler.get_total_steps(self.cfg.eval_epoch)
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        scaler = torch.cuda.amp.GradScaler() if self.cfg.enable_amp else None
        return scaler


@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def build_train_loader(self):
        from pointcept.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
        )
        self.comm_info["iter_per_epoch"] = len(train_loader)
        return train_loader
