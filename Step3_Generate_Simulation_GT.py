
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:25:57 2024

@author: Ziwei Zhang
"""

import os
import numpy as np
import pandas as pd
import math


pathGT = r'D:\DHPSFU\Data\GT'
pathSim = r"D:\DHPSFU\Data\Loc15\Simulation"

sim_list=list(range(1,501)) # Number of frames in the beads TIFF stack.

#%%

# Get a list of all files with the keyword in the path
def getListFiles(path, kwd = ''):
    filelist = [] 
    for root, dirs, files in os.walk(path):  
        for filespath in files: 
            if kwd in filespath or kwd in root:
                filelist.append(os.path.join(root,filespath)) 
    if len(filelist)==0:
        print('Error: There is no file containing "'+ kwd +'" under this path.')
    return filelist

def read_sim_info_data(file_name):
    cols = ['StackFrame', 'Bead_imID', 'BeadID', 'OriginalBeadFrame', 'y_start','y_end','x_start','x_end']      # useful column names
    sim_data = pd.read_csv(file_name, sep = ',', header = 0, engine='python')
   
    all_data = sim_data[cols]                                # take useful data
    num_locs = sim_data.shape[0]
    stackframe = sim_data['StackFrame'].copy()
    BeadID = sim_data['BeadID'].copy()
    OriginalBeadFrame = sim_data['OriginalBeadFrame'].copy()
    y_start = sim_data['y_start'].copy() 
    x_start = sim_data['x_start'].copy()
    return all_data, num_locs, stackframe, BeadID, OriginalBeadFrame, y_start, x_start

def read_gt_data(file_name):
    cols = ['Frame', 'X', 'Y', 'Z']      # useful column names
    GT_data = pd.read_csv(file_name, sep = ',', header =0, engine='python')
    all_data = GT_data[cols]                                # take useful data
    num_locs = GT_data.shape[0]
    frame = GT_data['Frame'].copy()
    x = GT_data['X'].copy()
    y = GT_data['Y'].copy()
    z = GT_data['Z'].copy() 
    return all_data, num_locs, frame, x, y, z

def read_3d_data(file_name):
    cols = ['X', 'Y', 'Z','Intensity','Frame']      # useful column names
    GT_data = pd.read_csv(file_name, sep = '\t', header =None, engine='python')
    GT_data.columns = cols
    #all_data = GT_data[cols]                                # take useful data
    num_locs = GT_data.shape[0]
    frame = GT_data['Frame'].copy()
    x = GT_data['X'].copy()
    y = GT_data['Y'].copy()
    z = GT_data['Z'].copy() 
    return GT_data, x, y, z, frame

def DH_calibration(file_name,sim_list):
    all_data, num_locs, frame, x, y, intensity, uncertainty = read_sim_info_data(file_name)  
    frame = all_data['#Frame']
    # take data from dataframe for analysis
    true_frame_list = list(frame.drop_duplicates())
    true_frames = np.asarray(sim_list)
    avg_x_all = np.zeros(len(sim_list))
    avg_y_all = avg_x_all.copy()
    avg_z_all = avg_x_all.copy()
    angle_all = avg_x_all.copy()
    
    # Calculate distance, relative intensity, average x, y, average intensity, and angles
    for i in range(len(sim_list)):
        one_frame = sim_list[i]
        index_frame = list(frame[frame==one_frame].index)
        fm1 = int(index_frame[0])
        fm2 = int(index_frame[1])
        x1 = x[fm1]
        x2 = x[fm2]
        y1 = y[fm1]
        y2 = y[fm2]
        avg_x = (x1+x2)/2
        avg_y = (y1+y2)/2
        avg_z = (one_frame-1)*33.3   # in PSFU
        angle = math.atan2(y2-y1, x2-x1)
        if angle<0:
            angle = angle+np.pi
        avg_x_all[i] = avg_x
        avg_y_all[i] = avg_y
        avg_z_all[i] = avg_z
        angle_all[i] = angle
        
        GT_x = np.mean(avg_x_all)
        GT_y = np.mean(avg_y_all)
    
    # Combine all data
    raw = pd.DataFrame({'Frame': true_frames, 'avg_x': avg_x_all, 
                        'avg_y': avg_y_all, 'avg_z':avg_z_all,
                        'Angle': angle_all})
   
    return raw, GT_x, GT_y
#%%

gt = getListFiles(pathGT,'.csv')
sim = getListFiles(pathSim,'simulate')


all_beads_gt=[]
for j in gt:
    all_data_gt, num_locs_gt, frame, x, y, z = read_gt_data(j)
    all_beads_gt.append(all_data_gt)

for i in sim:
    all_data_sim, num_locs_sim, stackframe, BeadID_sim, OriginalBeadFrame, y_start_sim, x_start_sim=read_sim_info_data(i)
    gt_X_list=[]
    gt_Y_list=[]
    gt_X_av=[]
    gt_Y_av=[]
    gt_Z_list=[]
    #gt_angle_list=[]
    for k in range(len(all_data_sim)):
        bead = all_data_sim['BeadID'][k]
        oribeadframe = OriginalBeadFrame[k]-1
        gt_X = x_start_sim[k]+all_beads_gt[bead]['X'][oribeadframe]
        gt_Y = y_start_sim[k]+all_beads_gt[bead]['Y'][oribeadframe]
        #gt_X = x_start_sim[k]
        #gt_Y = y_start_sim[k]
        gt_Z = (all_beads_gt[bead]['Z'][oribeadframe]+1.998)*1000    # 1.998 for Microscope 1, 2.46 for AP microscope 2
        #gt_angle = all_beads_gt[bead]['Angle'][oribeadframe]
        gt_X_list.append(gt_X)
        gt_Y_list.append(gt_Y)
        gt_Z_list.append(gt_Z)
        #gt_angle_list.append(gt_angle)
    
    all_data_sim['gt_X (nm)'] = gt_X_list
    all_data_sim['gt_Y (nm)'] = gt_Y_list
    all_data_sim['gt_Z (nm)'] = gt_Z_list

# Save the updated DataFrame to a CSV file
    saving_path = i[:-4]+'_GT_test.csv'
    all_data_sim.to_csv(saving_path, index=False)

print("Data saved to all_data_sim_with_gt.csv")
    
