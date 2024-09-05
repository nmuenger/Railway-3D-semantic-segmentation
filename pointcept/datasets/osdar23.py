"""
OSDaR Dataset

Original Author for semanticKITTI: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Adapted code Author: Nicolas Münger
Please cite our work if the code is helpful to you.
"""

import os
import sys
import numpy as np
import raillabel
from copy import deepcopy
from pathlib import Path
from torch.utils.data import Sampler
from railseg.pcd_processing import normalize_multisensor_intensity
import random
import copy
import pandas as pd

from .builder import DATASETS
from .defaults import DefaultDataset

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

@DATASETS.register_module()
class OSDaR23Dataset(DefaultDataset):
    def __init__(
        self,
        split="train",
        data_root="data/OSDaR_dataset",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        ignore_index=-1,
        names=None,
        split2seq=dict(
            train=["1_calibration_1.2","3_fire_site_3.1","3_fire_site_3.3","4_station_pedestrian_bridge_4.3","5_station_bergedorf_5.1","6_station_klein_flottbek_6.2","8_station_altona_8.1","8_station_altona_8.2","9_station_ruebenkamp_9.1","12_vegetation_steady_12.1","14_signals_station_14.1","15_construction_vehicle_15.1","20_vegetation_squirrel_20.1","21_station_wedel_21.1","21_station_wedel_21.2"],
            val=["2_station_berliner_tor_2.1","3_fire_site_3.4","4_station_pedestrian_bridge_4.2","4_station_pedestrian_bridge_4.5","6_station_klein_flottbek_6.1","7_approach_underground_station_7.2","9_station_ruebenkamp_9.3","9_station_ruebenkamp_9.4","9_station_ruebenkamp_9.5","9_station_ruebenkamp_9.7","11_main_station_11.1","14_signals_station_14.2","14_signals_station_14.3","18_vegetation_switch_18.1","21_station_wedel_21.3"],
            test=["1_calibration_1.1","3_fire_site_3.2","4_station_pedestrian_bridge_4.1","4_station_pedestrian_bridge_4.4","5_station_bergedorf_5.2","7_approach_underground_station_7.1","7_approach_underground_station_7.3","8_station_altona_8.3","9_station_ruebenkamp_9.2","9_station_ruebenkamp_9.6","10_station_suelldorf_10.1","13_station_ohlsdorf_13.1","16_under_bridge_16.1","17_signal_bridge_17.1","19_vegetation_curve_19.1"],
        ),
        learning_map=None,
        learning_map_inv=None, # Only necessary for unprocessed data
        annotation_disregarded=-2,
        use_preprocessed = True,
        return_track_poly3d = False,
        pose_dict = {"upright":0, "sitting":1, "lying":2, "other":3},
        csv_person_presence_path = "railseg/csv_stats/person_frames.csv",
        normalize_intensity = None
    ):
        self.ignore_index = ignore_index
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.names = names 
        self.split2seq = split2seq
        self.annotation_disregarded = annotation_disregarded
        self.use_preprocessed = use_preprocessed
        self.return_track_poly3d = return_track_poly3d
        self.pose_dict = pose_dict
        self.normalize_intensity = normalize_intensity

        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )
        self.person_data_idx = self.get_person_list(csv_person_presence_path) # The index of data item which have at least one person in it.

    def get_person_list(self, csv_person_presence_path):
        # Returns the indices which point to a frame where some pedestrian are located, for proper selection
        # of the second scene for copy pasting of pedestrian
        df = pd.read_csv(csv_person_presence_path,dtype=str) # This file contain the scenes/frame which have a person in it. Created in OSDaR_stats.ipynb
        if self.use_preprocessed:
            to_select = df['scene'] + "/" +  df['frame']
        else:
            to_select = df['scene'] + '/lidar/' + df['frame']
        person_data_idx= []
        for item in to_select: # For each element of the scenes containing a person, return  the index for that scene in data_list
            person_data_idx.extend([self.data_list.index(s) for s in self.data_list if item in s])
            
        return person_data_idx


        

    def get_data_list(self):
        split2seq = self.split2seq

        if isinstance(self.split, str):
            seq_list = split2seq[self.split]
        elif isinstance(self.split, list):
            seq_list = []
            for split in self.split:
                seq_list += split2seq[split]
        else:
            raise NotImplementedError

        data_list = []
        for seq in seq_list:
            seq_folder = os.path.join(self.data_root, seq)
            if self.use_preprocessed:
                seq_files = sorted(os.listdir(seq_folder))
                data_list += [os.path.join(seq_folder, file) for file in seq_files]
            else:
                seq_files = sorted(os.listdir(os.path.join(seq_folder, "lidar")))
                data_list += [os.path.join(seq_folder, "lidar", file) for file in seq_files]
        
        # Remove this specific frame which does not contain any anotation in the json file.
        problematic_frame = os.path.join(self.data_root, "15_construction_vehicle_15.1/lidar/075_1631531288.599807000.pcd")
        if problematic_frame in data_list:
            data_list.remove(problematic_frame)
       
        return data_list
    
    # --- UNprocessed ---
    def get_data_unprocessed(self, idx): 
        # idx is the index in the data_list that we want to retrieve. Note that if the eval is not run every epoch
        # this step is performed multiple time for a "single" frame and idx is therefore larger than len(self.data_list)
        
        data_path = Path(self.data_list[idx % len(self.data_list)])
        with open(data_path, "r") as b:
            scan = np.loadtxt(b, skiprows=11, usecols=(0,1,2,3,5)) # -> x|y|z|intensity|sensor_id
        
        coord = scan[:, :3]     # The x,y,z coordinates of the points 
        strength = scan[:, 3].reshape([-1, 1]) # The intensity of the points 
        sensor_id = scan[:, 4].reshape([-1, 1])

        if self.normalize_intensity:
            strength = normalize_multisensor_intensity(strength, sensor_id)

        scene_path = Path(data_path).parents[1] # Obtain .json file containing label for the current scene
        label_file = scene_path / (str(scene_path.name)+"_labels.json")
    
        segment = np.full(len(scan), 0) # Array containing the labels for each point ->first filled with the background tag
        current_instance = 0
        # Note: the instance attribute was created for my own purpose. In transform.py there is an InstanceParser which also utilize 
        # this attribute. I did not check how they use it, but probably the usage is similar
        instances = np.full(len(scan), current_instance) # Assign a number for each individual object -> first all filled with 0
        segment_person_status = np.full(len(scan), -1) # Only adding pose attribute if the point is of type "person", else: -1
        
        if os.path.exists(label_file):
            scene = raillabel.load(label_file) # Load json annotations for scene
            frame_nb = int(data_path.name.split('_')[0]) # returns the frame number as int '037'-> 37 
            scene_filtered = raillabel.filter(scene, include_frames=[frame_nb], include_annotation_types=['seg3d'])
            
            if frame_nb in scene_filtered.frames: # One of the point cloud doesnt have any frame annotation
                frame_objects = scene_filtered.frames[frame_nb].annotations.keys()
                
                for object in frame_objects:  
                    label_name = scene_filtered.frames[frame_nb].annotations[object].object.type # Object type name ('track','train'...)
                    label_number = self.learning_map[label_name] # Map name to tag ('person' -> 1, ...)

                    if label_number != self.annotation_disregarded: # Only change point attribute if it is desired
                        current_instance += 1
                        pts_idx = scene_filtered.frames[frame_nb].annotations[object].point_ids # Points index for the object
                        segment[pts_idx] = label_number   # Switch corresponding points with the class number
                        instances[pts_idx] = current_instance 

                        if label_name=="person":
                            pose = scene_filtered.frames[frame_nb].annotations[object].attributes["pose"]
                            segment_person_status[pts_idx] = self.pose_dict[pose]

            else:
                pass # Keep all points of the point cloud as ignore_index
            
            if self.return_track_poly3d:
                poly3d_filtered = raillabel.filter(scene, include_frames=[frame_nb], include_annotation_types=['poly3d'])
                dict_track = {}

                for annotation in poly3d_filtered.frames[frame_nb].annotations.keys():
                    track_name = poly3d_filtered.frames[frame_nb].annotations[annotation].object.name
                    if track_name not in dict_track:
                        dict_track[track_name] = {}
                    rail_side = poly3d_filtered.frames[frame_nb].annotations[annotation].attributes["railSide"]
                    dict_track[track_name][rail_side] = [point.asdict() for point in poly3d_filtered.frames[frame_nb].annotations[annotation].points]


        else:
            pass # If the file doesn't exist, no label point exist, we keep the segment filled with ignore_index
        
        # Create dict with for each point the coordinates, strength (intensity), segment (label) 
        data_dict = dict(coord=coord, strength=strength, segment=segment, instance=instances, person_pose=segment_person_status, scene=float(str(scene_path).rsplit("_",1)[1]), frame=frame_nb) # TODO if not used, remove the last two args.

        if self.return_track_poly3d:
            return data_dict, dict_track
        else:
            return data_dict

    # --- PREprocessed ---
    def get_data_preprocessed(self, idx): 
        # idx is the index in the data_list that we want to retrieve. Note that if the eval is not run every epoch
        # this step is performed multiple time for a "single" frame and idx is therefore larger than len(self.data_list)
        
        pcd_dir = self.data_list[idx % len(self.data_list)]
        scene_nb = float(pcd_dir.split("/")[-2].split("_")[-1])
        frame_nb = int(pcd_dir.split("/")[-1].split("_")[0])
        pcd = np.fromfile(self.data_list[idx % len(self.data_list)]).reshape([-1,8]) # X | Y | Z | intensity | classification | instance | person pose | sensor_id
        
        strength = pcd[:,3:4]
        sensor_id = pcd[:,7]

        if self.normalize_intensity:
            strength = normalize_multisensor_intensity(strength, sensor_id)

        # Note: the strength must be a vector in 2D
        data_dict = dict(coord=pcd[:,:3], strength=strength, segment=pcd[:,4].astype(int), instance=pcd[:,5].astype(int), person_pose=pcd[:,6].astype(int), scene=scene_nb, frame=frame_nb) # TODO if not used, remove the last two args.

        return data_dict
    
    def get_data(self, idx):
        if self.use_preprocessed:
            return self.get_data_preprocessed(idx)
        else:
            return self.get_data_unprocessed(idx)


    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"
        return data_name

    # The preparation of the train_data is changed from the original defaults.py
    def prepare_train_data(self, idx, transforms_list = None):
        data_dict = self.get_data(idx)

        if transforms_list is not None:
            data_dict["add_samples_transform"] = transforms_list # Add to the data dicrtionary the list of transforms which needs to be applied on this transform.
        else:
            data_dict["add_samples_transform"] = []

        # Only load second point cloud if required in transform process
        if ("PolarMixPaste" in [type(transform).__name__ for transform in self.transform.transforms]) or \
            ("PolarMixSwap" in [type(transform).__name__ for transform in self.transform.transforms]):
            #data_dict2 = self.get_data(np.random.randint(len(self.data_list)))
            data_dict2 = self.get_data(np.random.choice(self.person_data_idx)) # Actively select only scene which have persons in it for copy pasting
            data_dict2 = {k+"_2": v for k, v in data_dict2.items()}
            data_dict2["instance_2"] = data_dict2["instance_2"] + data_dict["instance"].max() # We want each instance to be individual, even between the two pcds
            data_dict = {**data_dict, **data_dict2}
            
        data_dict["pasting_successful"] = -1 # Add value to avoid issue in "Collect"
        data_dict = self.transform(data_dict)
 
        return data_dict
    
    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        result_dict = dict(
            segment=data_dict.pop("segment"), name=self.get_data_name(idx)
        )
        result_dict["entire_pcd_coord"] = data_dict["coord"] #Add the coordinates of the entire point cloud in the data dict
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part

        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = fragment_list
        return result_dict
    
    def __getitem__(self, idx): # Custom __getitem__ to handle two idx (+transform if given)
        if isinstance(idx, tuple): # If the AugmentedSampler is active, it return and index and a list of transform as a tuple
            idx, transform_list = idx
        elif isinstance(idx, int):
            transform_list = None
        else:
            sys.exit(f"Error: idx should only be of type 'int' or 'tuple', but received: {type(idx)}.")


        if self.test_mode:
            return self.prepare_test_data(idx) # The test loader doesn't need any transformation list
        else:
            return self.prepare_train_data(idx, transform_list)


class CustomSampler(Sampler):
    #TODO The augmented sampler does the same thing if no augmentation are applied. This could be removed.
    """Goal of this sampler is to force the batch to have at most ONE frame of either scene 20.1 or 12.1. Due to the large presence of vegetation in scene, 
        even after grid sampling, there is a large number of point in the sample"""
    def __init__(self, data, batch_size, augmentations=1):
        self.data = data
        self.dataset_size = data.__len__()
        self.batch_size = batch_size
        self.augmentations = augmentations
        # Getting the index for both problematic scenes
        self.scene_20_1_idx = [ind  for ind, string in enumerate(data.__dict__["data_list"]) if ("20_vegetation_squirrel_20.1" in string)]
        self.scene_12_1_idx = [ind  for ind, string in enumerate(data.__dict__["data_list"]) if ("12_vegetation_steady_12.1" in string)]
        self.non_prob_indices = [x for x in list(range(self.dataset_size)) if (x not in self.scene_20_1_idx) and (x not in self.scene_12_1_idx)]

    
    def shuffle_indices(self):
        # Create list of indices by starting the batch with the problematic pcd and completing it with non problematic pcd. Once all problematic
        # pcd have been seen, complete with the remaining indices. Note: this method is currently developed specifically for batch size of 2.
        indices = []
        non_prob_indices = copy.copy(self.non_prob_indices)
        
        for data_augment in range(self.augmentations): # Reshuffle and append as many time as desired
            random.shuffle(non_prob_indices) 
            
            i = 0
            for prob_idx in self.scene_12_1_idx+self.scene_20_1_idx:
                indices.append(prob_idx)
                indices.extend(non_prob_indices[i:i+self.batch_size-1])
                i += self.batch_size-1

            indices.extend(non_prob_indices[i:])
        return(indices)
    
    def __iter__(self):
        
        self.indices = self.shuffle_indices()

        return iter(self.indices)

    def __len__(self):
        return self.dataset_size


class AugmentedSampler(Sampler):
    """ Goal is to create additional samples seen during one epoch. The sample are artificially created by applying transform on existing data samples
        Instead of only returning the indice of the data to select, it gives a tuple containing the index AND the list of transforms that must be applied to it
        It also force the batches to have at most ONE frame of either scene 20.1 or 12.1. Due to the large presence of vegetation in scene, 
        even after grid sampling, there is a large number of point in the sample."""
    def __init__(self, data, cfg):
        # transform_list: list of dictionary containing the 
        self.data = data
        self.dataset_size = data.__len__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size_per_gpu#batch_size
        self.augmentations_list = cfg.augmented_sampler["augmentations_list"] #augmentations_list
        # Getting the index for both problematic scenes
        self.scene_20_1_idx = [ind  for ind, string in enumerate(data.__dict__["data_list"]) if ("20_vegetation_squirrel_20.1" in string)]
        self.scene_12_1_idx = [ind  for ind, string in enumerate(data.__dict__["data_list"]) if ("12_vegetation_steady_12.1" in string)]
        self.non_prob_indices = [x for x in list(range(self.dataset_size)) if (x not in self.scene_20_1_idx) and (x not in self.scene_12_1_idx)]
        self.drop_last_dataset_size = self.dataset_size-self.dataset_size%self.batch_size # Amount of samples when dropping the last indices contained in the "uncomplete" batch
        self.epoch = 0
    
    def shuffle_indices(self):
        # Create list of indices by starting the batch with the problematic pcd and completing it with non problematic pcd. Once all problematic
        # pcd have been seen, complete with the remaining indices. 
        # It then adds additional samples as defined in the config file.
        original_dataset_indices = []
        non_prob_indices = copy.copy(self.non_prob_indices)
        
        random.shuffle(non_prob_indices) 
        
        i = 0
        for prob_idx in self.scene_12_1_idx+self.scene_20_1_idx:
            original_dataset_indices.append(prob_idx)
            original_dataset_indices.extend(non_prob_indices[i:i+self.batch_size-1])
            i += self.batch_size-1

        original_dataset_indices.extend(non_prob_indices[i:])

        nb_samples_incomplete_batch = len(original_dataset_indices)%self.batch_size # Remove the indices which are in the last imcomplete batch, if there is any.

        if nb_samples_incomplete_batch > 0:
            del original_dataset_indices[-nb_samples_incomplete_batch:] # Makes code easier for consistency when selecting in the dataset with probabilities

        original_dataset_indices = [(x, []) for x in original_dataset_indices] # Transform list of indices to list of tuple (indices, transform) (In that case no transform)

        indices = original_dataset_indices

        for augmentation in self.augmentations_list:
            if ("start_epoch" not in augmentation.keys()) or (("start_epoch" in augmentation.keys()) and (self.epoch>=augmentation["start_epoch"])):
                self.verify_transforms_existence(augmentation["type"])
                #if (augmentation["augment_ratio"]>0) or (self.cfg.sweep): #TODO Look whether I want to leave this function in there (allow to launch a transformation after a certain number of epoch, without augmenting)
                augmented_indices = self.create_transformed_samples(original_dataset_indices, augmentation["augment_ratio"], augmentation["type"] )
                indices = indices + augmented_indices
                #else:
                    #indices = [(x[0], x[1]+augmentation["type"]) for x in indices]

        indices = self.shuffle_by_batch(indices, self.batch_size)

        if len(indices)!=self.__len__():
            sys.exit(f"Error: The length of the indices should match with the precomputed amount of samples in __len__(). Got {len(indices)} for len(indices) and {self.__len__()} for self.__len__().")

        print(f"{bcolors.OKCYAN}IMPORTANT INFO{bcolors.ENDC}: Original dataset size {self.dataset_size}, with batch_size {self.batch_size} and transform = {self.augmentations_list}. \nPredicted {self.__len__()} samples, and created list of indices of size {len(indices)}")
        return(indices)
    
    def create_transformed_samples(self, original_indices, augment_ratio, transforms_list):
        """ original_indices: list of indices created for the training split of the dataset
        augment_ratio: float between 0 and 1. Defines the proportion of the original dataset that we want to add on top
        transforms_list: list containing the transforms that will be applied on the samples, e.g. ["PolarMixPaste", "Sparsify"]
        returns: list containing for each added sample a tuple: (data_indices, transforms)
        I choose to augment dataset size by batches and not by samples, to avoid GPU memory over-usage.
        From the list of indices created before, we select by bucket the indices to add to dataset, with the corresponding augmentation """
        augmented_indices = []

        if (len(original_indices)%self.batch_size) != 0 : # Should not occur, here for debugging
            sys.exit("Error: The batch should all be 'complete' with all the same batch size.")

        nb_batch = int(len(original_indices)/self.batch_size) # Should anyway be a complete number, but np.random.choice accept only inter
        
        nb_batch_to_augment = int(augment_ratio*nb_batch)
        
        augmented_batch_idx = np.random.choice(nb_batch, nb_batch_to_augment, replace=False)
        for batch_idx in augmented_batch_idx:
            for i in range(self.batch_size):
                augmented_indices.append((original_indices[batch_idx*self.batch_size+i][0], transforms_list))
        
        
        return augmented_indices
    
    def shuffle_by_batch(self, indices, batch_size):
        # Shuffle list of indices while still preserving the indices which were put together in the same batch.
        batch_indices=int(len(indices)/batch_size)
        shuffled_batch_indices =np.random.choice(batch_indices, batch_indices, replace=False)
        shuffled_indices = []
        for batch_id in shuffled_batch_indices:
            shuffled_indices.extend(indices[batch_id*batch_size:batch_id*batch_size+batch_size])

        return shuffled_indices
    
    def verify_transforms_existence(self, transforms_list):
        # Check that the desired augmentation is properly defined in the list of transform in config.
        
        for desired_transform in transforms_list:
            desired_transform_found = False
            for defined_transform in self.cfg.data.train.transform: # Check presence in the overall defined transformations
                if defined_transform["type"]==desired_transform:
                    if "p" not in defined_transform.keys():
                        print(f"{bcolors.WARNING}WARNING{bcolors.ENDC}: probability was not found for transform '{desired_transform}', and default probability will be applied.\nDefine the probability in config to avoid unexpected behaviour.")
                    elif defined_transform["p"]>0:
                        print(f"{bcolors.WARNING}WARNING{bcolors.ENDC}: probability for transform '{desired_transform}' is bigger than 0. This means the transformation will also sometimes be applied on the regular dataset, not only on added samples.\nSet p=0 if you only intended to apply the transformation to added samples.")
                    else:
                        pass # The transformation is properly defined

                    desired_transform_found = True    
                    break
                        
            if desired_transform_found==False:
                # The desired transformation was not found in the overall list of defined transformation
                sys.exit(f"ERROR: the desired transform '{desired_transform}' was not found in the defined training transform from the config. Please ensure that it is properly defined.")
            
        return True
    
    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_total_steps(self, total_nb_epoch):
        #  The number of steps for one epoch is the total number of dataset sample divided by batch size.

        total_steps = 0
        for epoch in range(total_nb_epoch):
            total_steps += int(self.sampler_len_given_epoch(epoch)/self.batch_size)

        return total_steps

    def sampler_len_given_epoch(self, epoch):
        nb_added_samples = 0

        for augmentation in self.augmentations_list:
            if ("start_epoch" not in augmentation.keys()) or (("start_epoch" in augmentation.keys()) and (epoch>=augmentation["start_epoch"])):
                
                # added_samples = int(self.drop_last_dataset_size*augmentation["augment_ratio"])
                # batch_constrained_added_samples = added_samples - added_samples%self.batch_size # Because the added sample are randomly in batches, need to be taken into account
                # nb_added_samples += batch_constrained_added_samples

                # Compute the number of samples which will be produced by augmentation, following same logic as in create_transformed_samples
                nb_added_samples += int(int(self.drop_last_dataset_size/self.batch_size)*augmentation["augment_ratio"])*self.batch_size

            if ("start_epoch" in augmentation.keys()) and (epoch>=augmentation["start_epoch"]):
                print(f"{bcolors.WARNING}WARNING{bcolors.ENDC}: Some augmentation are activated only after a certain number of epoch. The handling of this by the scheduler was not precisely tested and unexpected behaviour might occur.")

        return self.drop_last_dataset_size+nb_added_samples

    def __iter__(self):
        
        self.indices = self.shuffle_indices()

        return iter(self.indices)

    def __len__(self):
        
        return self.sampler_len_given_epoch(self.epoch)

