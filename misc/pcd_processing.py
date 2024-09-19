import numpy as np
import pandas as pd
import torch
import laspy
import os
import argparse
# The order for the list of labels is defined by the order in the Labeling guide

osdar23_labels = ['background'
                  'person', 
                  'crowd',
                  'train',
                  'wagons',
                  'bicycle',
                  'group_of_bicycles',
                  'motorcycle',
                  'road_vehicle',
                  'animal',
                  'group_of_animals',
                  'wheelchair',
                  'drag_shoe', 
                  'track',
                  'transition', 
                  'switch',
                  'catenary_pole',
                  'signal_pole', 
                  'signal', 
                  'signal_bridge', 
                  'buffer_stop', 
                  'flame', 
                  'smoke']


def labels_name_to_number(df_path, labels_array):
    """Returns an array with the proper label numbers coresponding to the label name
        df_path: the path to the dataframe containing the matching information. Should have at least a column
                 called 'label_name' and one 'label_number'
        labels_array: array containing the name label for each of the point"""
    df = pd.read_csv(df_path)
    df_labels = pd.DataFrame({'label_name': labels_array})
    
    return df_labels.merge(df, how='left', on='label_name')[['label_number']].to_numpy().flatten()


def labels_to_numpy_rgb(labels_list):
    df = pd.read_csv('/workspaces/baseline/exp/labeling_osdar.csv')
    df_labels = pd.DataFrame({'label_name': labels_list})
    
    return df_labels.merge(df, how='left', on='label_name')[['red','green','blue']].to_numpy()

def retrieve_colors(labels_array, colors):
    """labels_array: 1D np.array containing the label number for each point
       colors: list of list containing RGB colors for each index"""
    np_colors = np.array(colors) # Convert list of list to numpy
    return np_colors[labels_array] # For each label number, return the associated color




def pcd_to_las(coord, export_path, gt_segm = None, pred_segm = None, intensity = None):
    """Creates a LAS file with the provided field.
    coord: [Nx3], np.array, the coordinates of the points
    export_path: file directory for export
    gt_segm: [Nx1], np.array, the ground truth segmentation of the pcd
    pred_segm: [Nx1], np.array, the infered segmentation of the pcd
    intensity: [Nx1], np.array, intensity attribute"""
    # Write info in las file
    # 1. Create a new header
    header = laspy.LasHeader(point_format=0) #https://laspy.readthedocs.io/en/latest/examples.html#creating-from-scratch
    
    # 2. Create a Las
    outfile = laspy.LasData(header)

    outfile.x = coord[:,0]
    outfile.y = coord[:,1]
    outfile.z = coord[:,2]

    if gt_segm is not None: # If provided, save GT segmentation under "classification" field
        if max(gt_segm)>31:
            print(f"The number of classes to save in the point cloud is above 31, which is the maximum that the LAS format accept. \
                  This point cloud will not be saved. It was supposed to go in the directory {export_path} ")
            return()
        else:
            outfile.classification = gt_segm
        
    if pred_segm is not None:  # If provided, save infered segmentation under "User Data" field
        outfile.user_data = pred_segm

    if (gt_segm is not None) and (pred_segm is not None): # If both are provided, save point where difference occurs under Withheld field
        outfile.withheld = (~np.equal(pred_segm, gt_segm)).astype(int) # Points with flag==1 indicate wrong prediction compared to GT
    
    if intensity is not None:
        outfile.intensity = intensity
    
    outfile.write(export_path)


def osdar_bin_to_las(bin_path, out_path):

    pcd = np.fromfile(bin_path).reshape([-1,8]) # X | Y | Z | intensity | classification | instance | person pose

    coord=pcd[:,:3]
    strength=pcd[:,3:4]
    segment=pcd[:,4].astype(int)
    sensor_id=pcd[:,7]

    strength = normalize_multisensor_intensity(strength, sensor_id)

    pcd_to_las(coord,out_path,segment)

def normalize_multisensor_intensity(intensity, sensor_id):
    # sensor_id = 0 -> Medium range, HesaiTech Pandar64 -> Divide by 255
    # sensor_id = 1,2,3 -> Long range, Livox Tele-15 -> Divide by 255
    # sensor_id = 4,5 -> Short range, Waymo Honeycomb -> tanh()

    index_medium_long_range = np.isin(sensor_id, test_elements=[0,1,2,3])
    index_short_range = np.isin(sensor_id, test_elements=[4,5])
    
    intensity[index_medium_long_range] = intensity[index_medium_long_range]/255
    intensity[index_short_range] = np.tanh(intensity[index_short_range])

    return intensity

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-bin_path", help="Path to preprocessed binary pcd")
    parser.add_argument("-export_path", help="Export path where pcd will be stored")
    args = parser.parse_args()
    binary_path = args.bin_path
    export_path = args.export_path
    osdar_bin_to_las(binary_path, export_path)