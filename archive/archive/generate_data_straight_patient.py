#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:08:18 2023

@author: jenskammerer
"""

import h5py
import scipy
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
import os
import pickle
from tqdm.auto import tqdm
from scipy.interpolate import interp1d
import torch 
from skimage.transform import downscale_local_mean
from medpy.io import load, header, save
from scipy.ndimage import rotate
#ReferenceCT
#images have to be of maximum size of 268 --> take DeepBackProj refCTs and crop, make sure nothing inside inscribed circle 

patients = ['male1', 'female1', 'male2','female2','male3', 'female3', 'male4','female4', 'male5', 'female5']

#refCT
#not necessary, can re-use from DBP

#slices
# same as for DBP

def read_refCT(patient):
    CT = np.load(r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/20231016_refCTs/'+patient+'_referenceCT.npy')
    return CT 

def read_refCTmask(patient):
    mask = np.load(r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/20231016_refCTs/'+patient+'_CTmask.npy')
    return mask 

# generate_slices()
def generate_sysm(nAngles):
    
    train_patients = ['VL053', 'SA659', 'PW341', 'HA229', 'GA795', 'FK619','WA653',  'HB341', 'MO098','MV679']
    test_patients = ['MH488']
    patients = ['VL053', 'SA659', 'PW341', 'HA229', 'GA795', 'FK619','WA653', 'MH488', 'HB341', 'MO098','MV679']

    MagnDeviation = 0.05
    Error = 'mixed'
    Angles = np.linspace(0,180, nAngles, endpoint = False)
    print(Angles)
    for patient in patients:
        print(patient)
        if patient in test_patients: 
            test_slices = np.load(r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/20231016_refCTs/'+patient+'_test_slices_han_pat.npy')#counting from 1
            pat_slices = [test_slices]
        else:
            train_slices = np.load(r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/20231016_refCTs/'+patient+'_train_slices_han_pat.npy')
            val_slices = np.load(r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/20231016_refCTs/'+patient+'_val_slices_han_pat.npy')
            pat_slices = [train_slices, val_slices]
        
        # mat5_folder =  r'/project/med5/IONCT/high_art/PHANTOM_INES/XCAT_Ines_'+patient+'_1mm1mm3mm' #
        # nSliceBlock = 7
        # npix = 256
        # n_ions = 100
        
        # mat5_folder =  r'/project/med5/IONCT/high_art/PHANTOM_INES/XCAT_Ines_'+patient+'_2mm' #
        # nSliceBlock = 13
        # npix = 128
        # n_ions = 100
        
        nSliceBlock = 1
        n_ions = 1
        path = r'/project/med2/Ines.Butz/Data/ML/PrimalDual/system_matrices/straight/'+patient+'_20231016/'#system matrices not changed, can use same path
        if not os.path.isdir(path):
            os.makedirs(path)
            
        CT = read_refCT(patient)
        mask = read_refCTmask(patient)
        print(mask.shape)
        
        sys_angles = []
        shape = [CT.shape[0], CT.shape[2]]
        for i, a in enumerate(Angles):
            system_matrix = []
            for p in np.arange(shape[0]):
                
                im0 = np.zeros(shape)
                theta = a 
                im_rot = im0
                im_rot[p,:] = 1
                im = rotate(im_rot, theta, reshape = False, order = 1)
                
                
                col = im.reshape(np.prod(shape), order = 'F')
                system_matrix  += [col]
            system_matrix = np.array(system_matrix).T
            system_matrix = scipy.sparse.csc_matrix(system_matrix)
            sys_angles += [system_matrix]
        
        for slices in pat_slices:
            #print(sorted(pat_slices[0]), sorted(pat_slices[1]))
            for s in tqdm(slices):
                CTblock = CT[:,s-math.floor(nSliceBlock/2)-1:s+math.floor(nSliceBlock/2),:]
                CTblock = CTblock.transpose(0,2,1)
                mask_image = mask[:,s-math.floor(nSliceBlock/2)-1:s+math.floor(nSliceBlock/2),:]
                mask_image = mask_image.transpose(0,2,1)
                print(mask_image.shape)
                mask_flat = mask_image.flatten(order = 'F')
                
            
                RSP_accurate = np.load(r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/calibs/'+patient+'_calibs_acc_'+str(int(100*MagnDeviation))+Error+r'/RSP_accurate_slice'+str(s)+'.npy')
                HU_original, RSP_original = np.load(r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/calibration_original.npy')
        
                ctArray = CTblock.flatten('F')
                ictArray = interp1d(HU_original, RSP_accurate, kind= 'linear')(ctArray)
                iCT = np.reshape(ictArray, np.shape(CTblock), order = 'F')
                iCTacc = iCT*mask_image
                print(CTblock.shape)
                for i, a in enumerate(Angles):
                    sys_m = sys_angles[i]
                    sys_m = sys_m.multiply(mask_flat[:, np.newaxis]).tocoo()
                    values = sys_m.data
                    indices = np.vstack((sys_m.row, sys_m.col))
                    i = torch.LongTensor(indices)
                    v = torch.FloatTensor(values)
                    shapeT = sys_m.shape
                    sys_coo_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shapeT))
                    torch.save(sys_coo_tensor, path+'/sysm_slice'+str(s)+'_angle'+str(int(a))+'.pt') #counting from 1
                    
                    summe = sys_m.sum(0)
                    ind0 = np.where(summe==0)[1]
                    summe[0,ind0] = 1
                    sys_norm =sys_m.multiply(1./summe)
                    
                    values = sys_norm.data
                    indices = np.vstack((sys_norm.row, sys_norm.col))
                    i = torch.LongTensor(indices)
                    v = torch.FloatTensor(values)
                    shapeT = sys_norm.shape
                    sys_coo_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shapeT))
                    torch.save(sys_coo_tensor.T, path+'/sysm_slice'+str(s)+'_angle'+str(int(a))+'_norm.pt')
                      
                    proj_angle= sys_norm.transpose().dot(iCTacc.flatten(order = 'F')).reshape(CTblock.shape[0],n_ions) #reshaped into matrix: pixels x ions (entry tracker projection image)
                    print(proj_angle.shape)
                    np.save(path+'/proj_slice'+str(s)+'_angle'+str(int(a))+'.npy', proj_angle)
                        
                  
generate_sysm(90)

#calibs stay the same
def compute_operator_norm(patient, s, a):
    path = r'/project/med2/Ines.Butz/Data/ML/PrimalDual/system_matrices/straight/'+patient+'_20231016/'#system matrices not changed, can use same path
    
    sys_coo_tensor  = torch.load(path+'/sysm_slice'+str(s)+'_angle'+str(int(a))+'.pt')
    sys_coo_tensor_norm  = torch.load(path+'/sysm_slice'+str(s)+'_angle'+str(int(a))+'_norm.pt')
    return torch.norm(sys_coo_tensor), torch.norm(sys_coo_tensor_norm)
def generate_norm(nAngles):    
    
    train_patients = ['VL053', 'SA659', 'PW341', 'HA229', 'GA795', 'FK619','WA653',  'HB341', 'MO098','MV679']
    test_patients = ['MH488']
    patients = ['VL053', 'SA659', 'PW341', 'HA229', 'GA795', 'FK619','WA653', 'MH488', 'HB341', 'MO098','MV679']
    Angles = np.linspace(0,180, nAngles, endpoint = False)
    forward_norms = []
    backward_norms = []
    for patient in patients:
        print(patient)
        if patient in test_patients: 
            test_slices = np.load(r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/20231016_refCTs/'+patient+'_test_slices_han_pat.npy')#counting from 1
            pat_slices = [test_slices]
        else:
            train_slices = np.load(r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/20231016_refCTs/'+patient+'_train_slices_han_pat.npy')
            val_slices = np.load(r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/20231016_refCTs/'+patient+'_val_slices_han_pat.npy')
            pat_slices = [train_slices, val_slices]
        
        for slices in pat_slices:
            for s in tqdm(slices):
                for i, a in enumerate(Angles):
                    norm1, norm2 = compute_operator_norm(patient, s, a)
                    forward_norms += [norm2.numpy()]
                    backward_norms += [norm1.numpy()]
    np.save(r'/project/med2/Ines.Butz/Data/ML/PrimalDual/patient_nAngles'+str(nAngles)+'_opNorms.npy',[np.mean(forward_norms), np.mean(backward_norms)])
generate_norm(90)


# patient = 'MH488'
# path = r'/project/med2/Ines.Butz/Data/ML/PrimalDual/system_matrices/straight/'+patient+'_20231016/'#system matrices not changed, can use same path
# sysm = torch.load(path+'sysm_slice76_angle90.pt')
# print(sysm.shape)
# img = torch.sparse.sum(sysm, dim=1).to_dense().numpy().reshape(256,256, order = 'F')
# print(img.shape)
# plt.figure()
# plt.imshow(img)
# plt.show()