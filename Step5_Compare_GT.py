# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 23:44:07 2024

@author: Ziwei Zhang
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import os


gt_folder = r"D:\DHPSFU\2025-02-04_Final\Loc10\GT"
data_folder = r"D:\DHPSFU\2025-02-04_Final\Loc10\Poly20"

wrong_loc_threshold_xy = 1*200
wrong_loc_threshold_z = 2*200

#%% 
def getListFiles(path, kwd = ''):
    filelist = [] 
    for root, dirs, files in os.walk(path):  
        for filespath in files: 
            if kwd in filespath or kwd in root:
                filelist.append(os.path.join(root,filespath)) 
    if len(filelist)==0:
        print('Error: There is no file containing "'+ kwd +'" under this path.')
    return filelist

def compare_results(gt_name, data_name, wrong_loc_threshold_xy, wrong_loc_threshold_z):
    
    gt_file = pd.read_csv(gt_name)
    gt_data = np.asarray(gt_file[['StackFrame', 'gt_X (nm)', 'gt_Y (nm)', 'gt_Z (nm)']]).astype(dtype=np.float64)
    data_file = pd.read_csv(data_name, header = None, sep='\t')
    Data_3d = np.asarray(data_file[[4, 0, 1, 2, 3]])
    Data_3d[:, 0] = Data_3d[:,0] - 1
    
    num_frame = int(np.max(gt_data[:,0]))
    
    result = []
    empty_arr = np.empty((1, 5), dtype = np.float64)
    empty_arr[:] = np.nan
    
    bad_frames = []
    mismatch_frames = []
    wrong_frames = []

    for i in range(num_frame+1):
        gt_i = gt_data[gt_data[:,0]==i] # for frame number i
        data_i=Data_3d[Data_3d[:,0]==i] # for frame number i
        if len(data_i) == 0: # bad frame with no loc in the data
            bad_frames.append(i) # record frame
            for j in range(len(gt_i)):
                result.append(empty_arr) # result append nan
        else:
            pairwise_dist = pairwise_distances(gt_i[:,1:], data_i[:,1:4]) # calculate pairwise distance between all locs in gt and all locs in data
            data_id = np.argmin(pairwise_dist, axis = 1) # get index of loc in pred that match gt       
            if len(np.unique(data_id)) == len(data_id): # if each gt loc matches a unique data loc
                for j in range(len(data_id)):
                    result.append(data_i[data_id[j],:].reshape((1, len(data_i[data_id[j],:]))))
            else:
                unique_id = np.unique(data_id)
                gts = []
                for k in unique_id:
                    gt_id = list(np.where(data_id==k)[0])
                    gt_dist = [pairwise_dist[g, k] for g in gt_id]
                    gt_min = np.argmin(gt_dist) # index of the gt loc with min distance for k in unique id
                    real_gt_id = gt_id[gt_min]
                    gts.append(real_gt_id)
                for g in range(len(data_id)):
                    if g in gts:
                        result.append(data_i[data_id[g],:].reshape((1, len(data_i[data_id[g],:]))))
                    else:
                        result.append(empty_arr)
                mismatch_frames.append(i)
    
    original_values = []
    
    result_df = pd.DataFrame(np.concatenate(result, axis=0))
    result_df.columns = ['frame', 'x', 'y', 'z', 'Intensity']
    combined = pd.concat((gt_file, result_df), axis = 1)
    combined['x_diff'] = np.abs(combined['x'] - combined['gt_X (nm)'])*100
    combined['y_diff'] = np.abs(combined['y'] - combined['gt_Y (nm)'])*100
    combined['z_diff'] = np.abs(combined['z'] - combined['gt_Z (nm)'])
    
    for idx, row in combined.iterrows():
        if (row['x_diff'] > wrong_loc_threshold_xy or 
            row['y_diff'] > wrong_loc_threshold_xy or 
            row['z_diff'] > wrong_loc_threshold_z):
            original_values.append(row)
            wrong_frames.append(row['frame'])  # Append frame number to mismatch_frames
            combined.loc[idx, ['frame','x', 'y', 'z', 'x_diff', 'y_diff', 'z_diff','Intensity']] = np.nan
    
    original_df = pd.DataFrame(original_values)
    original_df = original_df.drop(['StackFrame'], axis=1)
    combined.to_csv(data_name[:-3]+'_withData.csv', index = None)
    combined_drop = combined.dropna()
    combined_drop['3D_dist'] = (combined_drop['x_diff']**2 + combined_drop['y_diff']**2 + combined_drop['z_diff']**2)**0.5
    combined_drop.to_csv(data_name[:-3]+'_withData_noNAN.csv', index = None)
    # if len(bad_frames)!=0:
    #     bad_frames_df = pd.DataFrame(bad_frames)
    #     bad_frames_df.to_csv(gt_name[:-4]+'_bad_frames.csv', index = None, header = None)
    # if len(mismatch_frames)!=0:
    #     mismatch_frames_df = pd.DataFrame(mismatch_frames)
    #     mismatch_frames_df.to_csv(gt_name[:-4]+'_mismatch_frames.csv', index = None, header = None)
    
    max_length = max(len(bad_frames), len(mismatch_frames), len(wrong_frames), len(original_values))
    
    # Pad lists with None values if needed to ensure same length
    bad_frames += [None] * (max_length - len(bad_frames))
    mismatch_frames += [None] * (max_length - len(mismatch_frames))
    wrong_frames += [None] * (max_length - len(wrong_frames))
    
    # Create the combined frames DataFrame
    combined_frames = pd.DataFrame({
        'bad_frames': bad_frames,
        'mismatch_frames': mismatch_frames,
        'wrong_frames': wrong_frames
    })
    
    # Combine the original frames data with the other columns
    combined_frames = pd.concat([combined_frames, original_df.reset_index(drop=True)], axis=1)
    
    # Save the combined DataFrame to CSV
    if not combined_frames.empty:
        combined_frames.to_csv(data_name[:-3] + '_combined_frames.csv', index=False, header=True)
        
        
        
    avg_x_diff = combined_drop['x_diff'].mean()
    avg_y_diff = combined_drop['y_diff'].mean()
    avg_z_diff = combined_drop['z_diff'].mean()
    mean_3d_diff = combined_drop['3D_dist'].mean()
    median_x_diff = combined_drop['x_diff'].median()
    median_y_diff = combined_drop['y_diff'].median()
    median_z_diff = combined_drop['z_diff'].median()
    median_3d_diff = combined_drop['3D_dist'].median()
    no_loc_GT = gt_file.shape[0]
    no_loc_data = data_file.shape[0]
    no_loc_filter = combined_drop.shape[0]
    # Create a summary log
    log_content = f"Analysis Summary Log:\n\n"
    log_content += f"Mean X precisions (nm): {avg_x_diff:.4f}\n"
    log_content += f"Mean Y precisions (nm): {avg_y_diff:.4f}\n"
    log_content += f"Mean Z precisions (nm): {avg_z_diff:.4f}\n"
    log_content += f"Mean 3D precisions (nm): {mean_3d_diff:.4f}\n"
    log_content += f"{median_x_diff:.4f}\n"
    log_content += f"{median_y_diff:.4f}\n"
    log_content += f"{median_z_diff:.4f}\n"
    log_content += f"{median_3d_diff:.4f}\n"
    log_content += f"Median xyz and 3D precisions (nm)\n"
    log_content += f"Total no. of Locs in Simulation: {no_loc_GT:.0f}\n"
    log_content += f"{no_loc_data:.0f}\n"
    log_content += f"{no_loc_filter:.0f}\n"
    log_content += f"No. of loc detected/filtered by DHPSFU\n"
    log_content += f"Filter threshold xy (nm): {wrong_loc_threshold_xy:.1f}\n"
    log_content += f"Filter threshold z (nm): {wrong_loc_threshold_z:.1f}\n"
    # Write the summary to a text file
    log_filename = data_name[:-3]+'_analysis_log.txt'
    with open(log_filename, 'w') as f:
        f.write(log_content)
    
    print(f"Analysis log saved as {log_filename}")
    
#%% 
gt_name = getListFiles(gt_folder,'GT')
data_name = getListFiles(data_folder,'.3d')

for i in range(len(gt_name)):
    compare_results(gt_name[i], data_name[i], wrong_loc_threshold_xy, wrong_loc_threshold_z)