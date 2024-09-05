import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import copy
import random
import time 
import pandas as pd
import plotly.express as px
sys.path.insert(1, '/workspaces/baseline')
from pointcept.datasets.osdar23 import OSDaR23Dataset
from railseg.pcd_processing import pcd_to_las
from pointcept.datasets.transform import PolarMixPaste, Sparsify


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

train=["1_calibration_1.2","3_fire_site_3.1","3_fire_site_3.3","4_station_pedestrian_bridge_4.3","5_station_bergedorf_5.1","6_station_klein_flottbek_6.2","8_station_altona_8.1","8_station_altona_8.2","9_station_ruebenkamp_9.1","12_vegetation_steady_12.1","14_signals_station_14.1","15_construction_vehicle_15.1","20_vegetation_squirrel_20.1","21_station_wedel_21.1","21_station_wedel_21.2"]
val=["2_station_berliner_tor_2.1","3_fire_site_3.4","4_station_pedestrian_bridge_4.2","4_station_pedestrian_bridge_4.5","6_station_klein_flottbek_6.1","7_approach_underground_station_7.2","9_station_ruebenkamp_9.3","9_station_ruebenkamp_9.4","9_station_ruebenkamp_9.5","9_station_ruebenkamp_9.7","11_main_station_11.1","14_signals_station_14.2","14_signals_station_14.3","18_vegetation_switch_18.1","21_station_wedel_21.3"]
test=["1_calibration_1.1","3_fire_site_3.2","4_station_pedestrian_bridge_4.1","4_station_pedestrian_bridge_4.4","5_station_bergedorf_5.2","7_approach_underground_station_7.1","7_approach_underground_station_7.3","8_station_altona_8.3","9_station_ruebenkamp_9.2","9_station_ruebenkamp_9.6","10_station_suelldorf_10.1","13_station_ohlsdorf_13.1","16_under_bridge_16.1","17_signal_bridge_17.1","19_vegetation_curve_19.1"]

class person_stats:
    def __init__(self, save_html=False, save_jpeg=False):
        self.person_instance_distance = []
        self.person_instance_nb_points = []
        self.person_instance_mean_intensity = []
        self.person_paste_bool = []
        self.scene = []
        self.save_plot_to_html = save_html
        self.save_plot_to_jpeg = save_jpeg
        

    def update(self, data_dict):
        person_instances = np.unique(data_dict["instance"][np.where(data_dict["segment"]==1)[0]])

        for inst in person_instances:
            pts_idx = np.where(data_dict["instance"]==inst)[0] # Points index for the object
            instance_coords = data_dict["coord"][pts_idx]
            x_min, y_min= instance_coords[:,0:2].min(axis=0)
            x_max, y_max= instance_coords[:,0:2].max(axis=0)
            center_x = x_min+(x_max-x_min)/2
            center_y = y_min+(y_max-y_min)/2
            center_distance = np.linalg.norm([center_x, center_y])
            #center_distance = np.linalg.norm(data_dict["coord"][pts_idx][:,:2].mean(axis=0))

            self.person_instance_distance.append(center_distance)
            self.person_instance_nb_points.append(len(pts_idx))
            self.person_instance_mean_intensity.append(data_dict["strength"][pts_idx].mean())
            pasted_bool = (inst >= data_dict["instance_2"].min())
            self.person_paste_bool.append(pasted_bool)
            self.scene.append(data_dict["scene"])


    def create_df(self):
        pedestrian_df = pd.DataFrame({"dist":self.person_instance_distance, 
                                "nb_points":self.person_instance_nb_points, 
                                "mean_intensity":self.person_instance_mean_intensity,
                                "pasted": self.person_paste_bool,
                                "scene": self.scene})
        return pedestrian_df
    
    def create_plot(self):
        df=self.create_df()
        bucket_size = 5
        # Step 1: Create distance buckets
        df['dist_bucket'] = (df['dist'] // bucket_size) * bucket_size

        # Step 2: Group by distance buckets and calculate the mean of nb_points and mean_intensity, and count the number of samples
        result = df.groupby(['dist_bucket', 'pasted']).agg({'dist': 'count'}).reset_index()

        result2 = df.groupby('dist_bucket').agg(mean_nb_points=('nb_points','mean'), 
                                                mean_intensity=("mean_intensity", "mean"),
                                                total_nb_points=("nb_points","sum")).reset_index()

        result3 = df.groupby(['dist_bucket', 'pasted']).agg({'nb_points': 'mean'}).reset_index()
        # Step 3: Rename the 'dist' column to 'sample_count' to reflect the number of samples
        result = result.rename(columns={'dist': 'sample_count'})

        fig=px.bar(result, "dist_bucket","sample_count", color="pasted", title="Number of samples per distance range (for Train split)")
        fig.update_layout(xaxis_title=f"Distance ranges [x:x+{bucket_size}]", yaxis_title="Number of samples present at that distance")

        fig2=px.bar(result2, "dist_bucket","mean_nb_points", title="Average number of points per person instance, <br>according to their planar distance to sensor (in Train split)")
        fig2.update_layout(xaxis_title=f"Distance ranges [x:x+{bucket_size}]", yaxis_title="Average number of point per person instance")
        
        fig3=px.bar(result2, "dist_bucket","total_nb_points", title="Average number of points per person instance, <br>according to their planar distance to sensor (in Train split)")
        fig3.update_layout(xaxis_title=f"Distance ranges [x:x+{bucket_size}]", yaxis_title="Average number of point per person instance")

        fig4=px.bar(result3, "dist_bucket","nb_points", color="pasted", title="Average number of points per person instance, <br>according to their planar distance to sensor (in Train split) <br>separated between pasterd vs. original")
        fig4.update_layout(xaxis_title=f"Distance ranges [x:x+{bucket_size}]", yaxis_title="Average number of point per person instance", xaxis=dict(range=[0, max(result["dist_bucket"]) + 1]))


        if self.save_plot_to_html:
            fig.write_html("/workspaces/baseline/exp/figures/person_nb_samples_after_augmentation.html")
            fig2.write_html("/workspaces/baseline/exp/figures/person_nb_pts_after_augmentation.html")
            fig3.write_html("/workspaces/baseline/exp/figures/person_total_nb_pts_after_augmentation.html")
            fig4.write_html("/workspaces/baseline/exp/figures/person_nb_points_per_instance.html")
          
        if self.save_plot_to_jpeg:
            fig.write_image("/workspaces/baseline/exp/figures/person_nb_samples_after_augmentation.jpeg",scale=3)
            fig2.write_image("/workspaces/baseline/exp/figures/person_nb_pts_after_augmentation.jpeg",scale=3)
            fig3.write_image("/workspaces/baseline/exp/figures/person_total_nb_pts_after_augmentation.jpeg",scale=3)
            fig4.write_image("/workspaces/baseline/exp/figures/person_nb_points_per_instance.jpeg", scale=3)

# ----------- Start of the code ------------
save_to_las = False
save_dir = "exp/visualise_transforms"
analyse_only_one = False
idx1 = 20
idx2 = 259
seed = 32
random.seed(seed)
np.random.seed(seed)

dataset = OSDaR23Dataset(data_root="/workspaces/baseline/exp/preprocessed_pcd", learning_map=learning_map)
transform = PolarMixPaste(p=1, put_to_back_prob=1, csv_stat_path="/workspaces/baseline/railseg/csv_stats/pedestrian_density_per_distance.csv")

data_list = dataset.get_data_list()

df = pd.read_csv("/workspaces/baseline/railseg/csv_stats/person_frames.csv",dtype=str) # This file contain the scenes/frame which have a person in it. Created in OSDaR_stats.ipynb

to_select = df['scene'] + "/" +  df['frame']
       
person_data_idx= []
for item in to_select: # For each element of the scenes containing a person, return  the index for that scene in data_list
    person_data_idx.extend([data_list.index(s) for s in data_list if item in s])
            
if analyse_only_one:
    frames_to_process = [idx1]
else:
    frames_to_process = range(len(data_list))

counter_sucess_augment = 0
start_time = time.time()
person_stats_tracker = person_stats(save_html=True)

debug_paste_distance_list=[]
debug_paste_nb_points_list=[]

for i in frames_to_process:
    
    data_dict = dataset.get_data(i) 
    data_dict2 = dataset.get_data(idx2)#dataset.get_data(np.random.choice(person_data_idx)) # Actively select only scene which have persons in it for copy pasting
    data_dict2 = {k+"_2": v for k, v in data_dict2.items()}
    data_dict2["instance_2"] = data_dict2["instance_2"] + data_dict["instance"].max() # We want each instance to be individual, even between the two pcds
    data_dict = {**data_dict, **data_dict2}

    if save_to_las:
        folder = os.path.join(save_dir, str(data_dict["scene"])+"_"+str(data_dict["frame"]))
        os.makedirs(folder, exist_ok=True)
        pcd_to_las(data_dict["coord"], os.path.join(folder, "pcd1.las"), data_dict["segment"])
        pcd_to_las(data_dict["coord_2"], os.path.join(folder, f'pcd2_{data_dict["scene_2"]}_{data_dict["frame_2"]}.las'), data_dict["segment_2"], pred_segm=data_dict["instance_2"])
    
    data_dict,  debug_paste_distance, debug_paste_nb_points = transform(data_dict)

    debug_paste_distance_list.extend(debug_paste_distance)
    debug_paste_nb_points_list.extend(debug_paste_nb_points)

    if save_to_las:
        pcd_to_las(data_dict["coord"], os.path.join(folder, "pcd_resulting.las"), data_dict["segment"], pred_segm=data_dict["instance"])


    person_stats_tracker.update(data_dict) 
    counter_sucess_augment+=data_dict["pasting_successful"]
    if i % 20 == 0:
        print(f"{i}/{len(data_list)}")
end_time = time.time()

# person_stats_tracker.create_plot()
# person_stats_tracker.create_df().to_csv("exp/figures/df_person_stats2.csv") # .create_plot()
# pd.DataFrame({"debug_nb_points":debug_paste_nb_points_list, "debug_distance":debug_paste_distance_list}).to_csv("exp/figures/debug_df_paste.csv")
print(f"Out of {len(data_list)} samples, {counter_sucess_augment} augmentation actually succeeded")

print(f"Took {end_time-start_time} sec to process one run through dataset")