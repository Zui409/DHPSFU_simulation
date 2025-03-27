# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 00:25:39 2022

@author: Ziwei Zhang
"""

import pandas as pd
import os
import win32com.client as win32

TS_folder = r'D:\DHPSFU\Data\Loc5\TS'
img_format = r'.csv'

pixel_size = 130
intensity_conversion_factor = 1 # intensity(thunderstorm) * factor = intensity(GDSC)
uncertainty_conversion_factor = 1 # uncertainty(thunderstorm) * factor = precision(GDSC)

# Get all files in the path
def getListFiles(path):
    filelist = [] 
    for root, dirs, files in os.walk(path):  
        for filespath in files: 
            filelist.append(os.path.join(root,filespath)) 
    return filelist

# Get all images with the specified format
def getimages(img_format, filelist):
    img_list = []
    for filename in filelist:
        if filename[-4:]==img_format:
            img_list.append(filename)
    return img_list

# Create folders for analysed data
def create_sub_folders(path):
    path_split = path.split('\\') # split the path name
    raw_img_folder = path_split[-1]
    dir_list = []
    new_dir_list = []
    for root, dirs, files in os.walk(path):
        dir_list.append(os.path.join(root))
    for dir_name in dir_list:
        new_dir_name = dir_name[:-len(raw_img_folder)] + r'Converted_GDSC'
        if not os.path.exists(new_dir_name):
            os.makedirs(new_dir_name)
        new_dir_list.append(new_dir_name)
    return dir_list, new_dir_list

# Create filenames for processed images
def create_processed_img_names(path, img_list):
    mask_list = []
    path_split = path.split('\\') # split the path name
    path_len = len(path_split[-1]) # name length of the last folder
    parent_path = path[0:-path_len] # find the parent folder of the image folder
    new_path = parent_path+r'Converted_GDSC'
    for img_name in img_list:
        img_old_name = img_name[len(path):]
        img_new_name1 = new_path+img_old_name
        img_name_split = img_new_name1.split('\\')
        img_filename = img_name_split[-1]
        mask_filename = r'GDSC_' + img_filename
        mask_name = img_new_name1.replace(img_filename, mask_filename)
        mask_list.append(mask_name)
    return new_path, mask_list

def read_TS(TS_name, pixel_size, intensity_conversion_factor, uncertainty_conversion_factor):
    TS_data = pd.read_csv(TS_name)
    num_locs = TS_data.shape[0]
    frame = TS_data['frame'].copy()
    x = TS_data['x [nm]'].copy() / pixel_size
    y = TS_data['y [nm]'].copy() / pixel_size
    intensity = TS_data['intensity [photon]'].copy() * intensity_conversion_factor
    uncertainty = TS_data['uncertainty [nm]'].copy() * uncertainty_conversion_factor
    return num_locs, frame, x, y, intensity, uncertainty

filelist = getListFiles(TS_folder)
TS_list = getimages(img_format, filelist)
dir_list, new_dir_list = create_sub_folders(TS_folder)
new_path, GDSC_list = create_processed_img_names(TS_folder, TS_list)

for i in range(0, len(TS_list)):
    TS_name = TS_list[i]
    num_locs, frame, x, y, intensity, uncertainty = read_TS(TS_name, pixel_size, intensity_conversion_factor, uncertainty_conversion_factor)
    GDSC_empty = pd.DataFrame(columns = ['#Frame', 'origX', 'origY', 'origValue', 'Error', 'Noise (photon)', 'Mean (photon)', 'Background (photon)', 'Intensity (photon)', 'X (px)', 'Y (px)', 'Z (px)', 'S (px)', 'Precision (nm)'])
    GDSC_empty['#Frame'] = frame.values
    GDSC_empty['X (px)'] = x.values
    GDSC_empty['Y (px)'] = y.values
    GDSC_empty['Intensity (photon)'] = intensity.values
    GDSC_empty['Precision (nm)'] = uncertainty.values
    GDSC_empty.fillna(0, inplace = True)
    GDSC_name = GDSC_list[i]
    GDSC_empty.to_csv(GDSC_name, index = False, encoding='utf-8')

for i in range(0, len(GDSC_list)):
    GDSC_name_1 = GDSC_list[i]
    xls_name = GDSC_name_1.replace('csv', 'xls')
    excel = win32.gencache.EnsureDispatch('Excel.Application')
    wb = excel.Workbooks.Open(GDSC_name)
    wb.SaveAs(xls_name, FileFormat = 56)
    wb.Close
    excel.Application.Quit()