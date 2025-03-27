# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:05:23 2024

@author: Ziwei Zhang
"""

import numpy as np
from skimage import io
from tifffile import imsave
import pandas as pd

calib_info_name = r"D:\DHPSFU\Data\Calib_info.csv"  # File containing the directory and info of the beads TIFF used for simulation. 

# Blank stack paras
canvas_size = 300
frame_num = 5000

num_dot_pf = 15 # Number of dots per frame
num_dot_var = False # if the number of dots vary from frame to frame
num_dot_sigma = 0.2 # sigma of the variation

bg_type = 'Gaussian' # Choose from Gaussian or Poisson
bg = [0,5000,10000,15000,20000]


bg_sigma = 0.2  # only useful when choose Gaussian
Int_sig = 0.2
modify_bead_range = [1, 1.2] # intensity variation range

tophat_r = 15
thres = 1
erode_size = 1


#%% Functions

def no_subtbg(calib_im, tophat_r, thres, erode_size):
    num_frame = calib_im.shape[0]
    out = []
    thress = []
    for i in range(num_frame):
        thress.append(0)
    out = calib_im
    return thress, out

def load_calib(calib_info_name, tophat_r, thres, erode_size):
    info = pd.read_csv(calib_info_name, sep = ',')
    beads = []
    bead_xy = []
    for i in range(len(info)):
        infoo = info.iloc[i]
        x_s = infoo['y_start']
        x_e = infoo['y_end']#+1
        y_s = infoo['x_start']
        y_e = infoo['x_end']#+1
        f_s = infoo['f_start']-1
        f_e = infoo['f_end']
        im = io.imread(infoo['FileName'])
        #thress, out = subtract_bg(im, tophat_r, thres, erode_size)
        thress, out = no_subtbg(im, tophat_r, thres, erode_size)
        bead = out[f_s:f_e, x_s:x_s+x_e, y_s:y_s+y_e]
        beads.append(bead)
        bx = x_e # x_size (number of rows)
        by = y_e # y_size (number of cols)
        bead_xy.append((bx, by))
    return info, beads, bead_xy

def choose_bead(info, beads, bead_xy, num_dot_pf, num_dot_var, num_dot_sigma, frame_num, canvas_size):
    num_lib = len(beads) # total number of beads in library
    b_frame = [b.shape[0] for b in beads] # number of frames in all bead stacks
    im_id = []
    for n in range(num_lib):
        for b in range(0, b_frame[n]):
            im_id.append(np.asarray([n, b]))
    im_id = pd.DataFrame(im_id)
    im_id.columns = ['BeadID', 'Frame'] # ID of all small images, including bead ID and index of frame
    
    # Generate number of dots on each frame of the simulated stack.
    # can choose from fixed number or variable number (num_dot_var = True or False)
    if num_dot_var == False:
        n_dot = num_dot_pf * np.ones((frame_num, 1))
    else:
        n_dot = np.random.normal(num_dot_pf, num_dot_pf*num_dot_sigma, frame_num)
        n_dot[n_dot<0] = 0
    n_dot = np.round(n_dot).astype(np.int16)
    
    # Randomly select images from the library
    id_f = []
    for i in range(frame_num):
        n = n_dot[i]
        id_f.append(np.random.randint(0, len(im_id), n))
    
    ns = []
    ids = []
    for i in range(frame_num):
        for j in range(len(id_f[i])):
            ns.append(i)
            ids.append(id_f[i][j])
    
    x1s = []
    x2s = []
    y1s = []
    y2s = []
    bead = []
    orif = []
    rawf = []
    for i in id_f:
        for j in i:
            im = im_id.loc[j]
            im_bead = im['BeadID']
            f_start = int(info[info['BeadID']==im_bead]['f_start'])
            im_orif = im['Frame']
            im_rawf = im['Frame'] + f_start
            
            bead_x = bead_xy[im_bead][0] # bead image x length in python
            bead_y = bead_xy[im_bead][1] # bead image y length in python
            
            posx = np.random.randint(0, canvas_size-bead_x, 1) # set position within the border
            posy = np.random.randint(0, canvas_size-bead_y, 1)
            
            x1s.append(posx[0])
            x2s.append(posx[0]+bead_x)
            y1s.append(posy[0])
            y2s.append(posy[0]+bead_y)
            bead.append(im_bead)
            orif.append(im_orif)
            rawf.append(im_rawf)
    
    table = pd.DataFrame([ns, ids, bead, orif, rawf, x1s, x2s, y1s, y2s]).T
    table.columns = ['StackFrame', 'Bead_imID', 'BeadID', 'BeadFrame', 'OriginalBeadFrame', 'x_start', 'x_end', 'y_start', 'y_end']
    # Warning: the xy coordination here is in python (x is row index, y is column index)
    return n_dot, table

def modify_bead(beads, modify_bead_range):
    mod_beads = []
    ints = []
    mod_by_frame_c = False
    for bead in beads:
        
        # increase the intensity of frames of lower than mean
        num_frame = bead.shape[0]
        avg_by_frame = [np.mean(bead[f,:,:]) for f in range(num_frame)] # average intensity of all frames of the bead
        mean_int = np.mean(avg_by_frame)
        mod_by_frame = []
        if mod_by_frame_c==True:
            for avg in avg_by_frame:
                if avg < mean_int:
                    mod = mean_int-(avg*avg/mean_int)+(avg*avg*avg/mean_int/mean_int)
                    mod_by_frame.append(mod/avg)
                else:
                    mod_by_frame.append(avg/avg)
        else:
            for avg in avg_by_frame:
                mod_by_frame.append(1)
        # introduce random variation
        variation = np.random.uniform(modify_bead_range[0], modify_bead_range[1], num_frame)
        mod_variation = mod_by_frame*variation
        mod_bead = []
        for f in range(num_frame):
            mod_frame = bead[f,:,:]*mod_variation[f]
            mod_bead.append(mod_frame)
        mod_beads.append(np.asarray(mod_bead))
        
        avg_int = np.asarray(avg_by_frame).reshape((num_frame, 1))
        int_fac = np.asarray(mod_by_frame).reshape((num_frame, 1))
        var_fac = np.asarray(variation).reshape((num_frame, 1))
        ints.append(np.concatenate((avg_int, int_fac, var_fac), axis = 1))
    
    ori_max_int = np.max([np.max(b) for b in beads])
    max_int = np.max([np.max(b) for b in mod_beads])
    
    nor_beads = []
    nor_int = []
    for mod_bead in mod_beads:
        nor = mod_bead/max_int*ori_max_int*modify_bead_range[1]
        nor = nor.astype(np.uint16)
        nor_beads.append(nor)
        nor_int.append([np.mean(nor[f,:,:]) for f in range(nor.shape[0])])
    
    int_summary = []
    for i in range(len(ints)):
        int_sum = pd.DataFrame(np.concatenate((ints[i], np.asarray(nor_int[i]).reshape((len(nor_int[i]), 1))), axis=1))
        int_sum.columns = ['original_mean_int', 'mean_fac', 'variation', 'final_int']
        int_summary.append(int_sum)

    return nor_beads, int_summary

def create_stack(canvas_size, frame_num, beads, table):
    stack = []
    for i in range(frame_num):
        canvas = np.zeros((canvas_size, canvas_size)).astype(np.uint16)
        bb = table[table['StackFrame']==i]
        for j in range(len(bb)):
            b_id = bb.iloc[j]['BeadID']
            b_fm = bb.iloc[j]['BeadFrame']
            x1 = bb.iloc[j]['x_start']
            x2 = bb.iloc[j]['x_end']
            y1 = bb.iloc[j]['y_start']
            y2 = bb.iloc[j]['y_end']
            sf = beads[b_id][b_fm]
            img = np.zeros((canvas_size, canvas_size)).astype(np.uint16)
            img[x1:x2, y1:y2] = sf
            canvas = canvas + img
        stack.append(canvas)
    stack = np.asarray(stack)
    return stack

def add_bg(stack, bg_noise, bg_sigma, bg_type):
    num_f = stack.shape[0]
    canv = (stack.shape[1], stack.shape[2])
    stack_noisy = []
    for j in range(num_f):
        if bg_type == 'Gaussian':
            noisy = np.random.normal(bg_noise, bg_sigma*bg_noise, canv).astype(np.uint16)
        elif bg_type == 'Poisson':
            noisy = np.random.poisson(bg_noise, canv).astype(np.uint16)
        img = stack[j,:,:]
        idx = (img<bg_noise*0.5) * (img>0)
        img[idx] = img[idx]# * 0.3
        #noisy[img>0] = 0
        stack_noisy.append(img+noisy)
    stack_noisy = np.asarray(stack_noisy)
    return stack_noisy

def save_img(stack, stack_noisy, table, calib_info_name, bg_noise, int_summary):
    path = calib_info_name[:calib_info_name.rfind('\\')]
    stack_n = path + '\\stack'+str(bg_noise)+'.tif'
    stack_noisy_n = path + '\\stack_noisy'+str(bg_noise)+'.tif'
    info_n = path + '\\simulate_info'+str(bg_noise)+'.csv'
    imsave(stack_n, stack) 
    imsave(stack_noisy_n, stack_noisy)
    table_save = table.copy()
    table_save.rename(columns={'x_start':'y_start', 'x_end':'y_end','y_start':'x_start','y_end':'x_end'}, inplace=True)
    table_save.drop(columns=['BeadFrame'], inplace = True)
    table_save.to_csv(info_n, index = None)
    bead_int_name = [path + '\\bead_intensity_'+str(i)+'bgnoise_+'+str(bg_noise)+'.csv' for i in range(len(int_summary))]
    for i in range(len(int_summary)):
        summary = int_summary[i]
        summary.to_csv(bead_int_name[i], index = None)
    
#%% Run
for i in range(len(bg)):    
    bg_noise = bg[i]
    info, beads, bead_xy = load_calib(calib_info_name, tophat_r, thres, erode_size)
    beads, int_summary = modify_bead(beads, modify_bead_range)
    n_dot, table = choose_bead(info, beads, bead_xy, num_dot_pf, num_dot_var, num_dot_sigma, frame_num, canvas_size)
    stack = create_stack(canvas_size, frame_num, beads, table)
    stack_noisy = add_bg(stack, bg_noise, bg_sigma, bg_type)
    save_img(stack, stack_noisy, table, calib_info_name, bg_noise, int_summary)