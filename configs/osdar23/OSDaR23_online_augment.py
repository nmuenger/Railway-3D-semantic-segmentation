_base_ = ["../_base_/default_runtime.py"]

# ---- misc custom setting ----------------
sweep=False
normalize_intensity=True
batch_size = 2 # bs: total bs in all gpus
mix_prob = 0. # From what I understand, some of the point cloud in a batch can be "mixed" given this probability (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9665916)
empty_cache = True
enable_amp = True
num_worker = 6
seed = 42
ignore_index = -1
specific_osdar23_sampler=False
augment_dataset_size = 1 # How many time to iterate through the whole dataset for one epoch
augmented_sampler = dict(active = True,
                         augmentations_list = [
                                               #dict(type=["PolarMixPaste"], augment_ratio=0.), # If there are 100 samples, with one augm. with augment_ratio=0.4 -> 140 samples
                                               ])
csv_person_presence_path="railseg/csv_stats/person_frames.csv"

# ---- Custom param for railseg --------------------------
pcd_to_wandb = True # If set to true, point cloud segmentation prediction will be uploaded to WandB
upload_pcd_every_x_epoch = 10 # The pcd will be uploaded every x epoch

# ---- Split to Seq. -------------------------------------
split2seq = dict(
            train=["1_calibration_1.2","3_fire_site_3.1","3_fire_site_3.3","4_station_pedestrian_bridge_4.3","5_station_bergedorf_5.1","6_station_klein_flottbek_6.2","8_station_altona_8.1","8_station_altona_8.2","9_station_ruebenkamp_9.1","12_vegetation_steady_12.1","14_signals_station_14.1","15_construction_vehicle_15.1","20_vegetation_squirrel_20.1","21_station_wedel_21.1","21_station_wedel_21.2"],
            val=["2_station_berliner_tor_2.1","3_fire_site_3.4","4_station_pedestrian_bridge_4.2","4_station_pedestrian_bridge_4.5","6_station_klein_flottbek_6.1","7_approach_underground_station_7.2","9_station_ruebenkamp_9.3","9_station_ruebenkamp_9.4","9_station_ruebenkamp_9.5","9_station_ruebenkamp_9.7","11_main_station_11.1","14_signals_station_14.2","14_signals_station_14.3","18_vegetation_switch_18.1","21_station_wedel_21.3"],
            test=["1_calibration_1.1","3_fire_site_3.2","4_station_pedestrian_bridge_4.1","4_station_pedestrian_bridge_4.4","5_station_bergedorf_5.2","7_approach_underground_station_7.1","7_approach_underground_station_7.3","8_station_altona_8.3","9_station_ruebenkamp_9.2","9_station_ruebenkamp_9.6","10_station_suelldorf_10.1","13_station_ohlsdorf_13.1","16_under_bridge_16.1","17_signal_bridge_17.1","19_vegetation_curve_19.1"],
        )

# ---- definition of dictionary with associated color ----
num_classes = 8 

#vv Class which are evaluated during the validation process. The training loss itself is computed on all class resulting from the Learning map vv
evaluated_class = [0, #background  
                    1, #person
                    2, #train
                    4, #track
                    5, #catenary pole
                    6, #signal
                    7] #buffer stop

# This attribute is for annotation which should actas if they don't actually exist, and not be actively mapped to background.
# It is mainly for the case of switch, which are supperposed with track annotation. If we map those to track, then
# the shape of the segmentation is not as precise, but if we map them to background, then possibility that there is a "hole" in that place in the track.
annotation_disregarded = -2 

learning_map = {
            'background':         0,
            'person':             1,
            'crowd':              1,
            'train':              2,
            'wagons':             2,
            'bicycle':            0,
            'group_of_bicycles':  0,
            'motorcycle':         0,
            'road_vehicle':       3,
            'animal':             0,
            'group_of_animals':   0,
            'wheelchair':         0,
            'drag_shoe':          0,
            'track':              4,
            'transition':         4,
            'switch':             annotation_disregarded,
            'catenary_pole':      5,
            'signal_pole':        6,
            'signal':             6,
            'signal_bridge':      0,
            'buffer_stop':        7,
            'flame':              0,
            'smoke':              0
        }

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


names = [
'background',
'person',       #1
'train' ,       #2
'road_vehicle', #3
'track',        #4
'catenary_pole',#5
'signal',       #6
'buffer_stop',  #7
]

# ----- model settings ------------------------------
model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=4,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash= True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=ignore_index),
    ],
)

# ---- scheduler settings -------------------------------------
epoch = 50
eval_epoch = 50
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=0.0001)]

# ---- dataset settings ------------------------------------
dataset_type = "OSDaR23Dataset"
data_root = "/workspaces/baseline/exp/preprocessed_pcd"

# ---- DATA PARAM -----------------------------------------
data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    split2seq=split2seq,
    learning_map=learning_map,
    learning_map_inv=learning_map_inv,
    color_map=color_map,
    names=names,
    evaluated_class=evaluated_class,


    # -------- Training specific --------------------------
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            #dict(type="PolarMixSwap", p=0.5),
            dict(type="PolarMixPaste", p=0, csv_stat_path="/workspaces/baseline/railseg/csv_stats/pedestrian_density_per_distance.csv"),
            dict(type="RandomRotate", angle=[-1/12, 1/12], axis="z", center=[0, 0, 0], p=0.5), # The angle is multiplied by pi
           
            dict(type="RandomScale", scale=[0.9, 1.1]),

            dict(type="RandomFlip", p=0.5, around_xaxis_only=True),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
 
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment", "instance"),
                return_grid_coord=True,
            ),

            dict(type="Sparsify", end_range=80, track_label=learning_map["track"], p=0), # start_range = lower bound of 10 meter section 
            #dict(type="SparsifyTrackIgnore", end_range=80, track_label=learning_map["track"], p=1),

            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "scene", "frame", "instance", "pasting_successful"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
        learning_map=learning_map,
        learning_map_inv=learning_map_inv,
        split2seq=split2seq,
        annotation_disregarded=annotation_disregarded,
        csv_person_presence_path = csv_person_presence_path,
        normalize_intensity=normalize_intensity
    ),
    # -------- Validation specific --------------------------
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            # dict(type="PointClip", point_cloud_range=(-51.2, -51.2, -4, 51.2, 51.2, 2.4)),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode='center'),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "scene", "frame"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
        learning_map=learning_map,
        learning_map_inv=learning_map_inv,
        split2seq=split2seq,
        annotation_disregarded=annotation_disregarded,
        normalize_intensity=normalize_intensity
    ),
    # -------- Test specific --------------------------
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        # transform=[
        #     dict(type="Copy", keys_dict={"segment": "origin_segment"}),
        #     dict(
        #         type="GridSample",
        #         grid_size=0.025,
        #         hash_type="fnv",
        #         mode="train",
        #         keys=("coord", "strength", "segment"),
        #         return_inverse=True,
        #     ),
        # ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                keys=("coord", "strength"),
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "strength"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [
                    dict(type="RandomScale", scale=[0.9, 0.9]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                    dict(type="RandomFlip", p=1),
                ],
                [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
                [
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.1, 1.1]),
                    dict(type="RandomFlip", p=1),
                ],
            ],
        ),
        ignore_index=ignore_index,
        names=names,
        split2seq=split2seq,
        learning_map=learning_map,
        learning_map_inv=learning_map_inv,
        annotation_disregarded=annotation_disregarded,
        normalize_intensity=normalize_intensity
    ),
)

# ---- hook --------------------------------------
# If the hooks are defined in this config, we can override the default
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None, save_best_at=None),
    dict(type="CUDAMemoryConsumption"),
    dict(type="SuccessfulPastingCounter"),
    # dict(type="PreciseEvaluator", test_last=False),
]