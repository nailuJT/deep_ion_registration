#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 12:24:54 2023

@author: ines.butz
"""

import numpy as np
import torch
from scipy.interpolate import interp1d
import pickle
import math

class DatasetPrimalDual(torch.utils.data.Dataset):
    #Test when data available! 
    def __init__(self, datalist, n_primal,  norm, op = 'train', pretrain = False):
        
        """Constructor.

        datalist: paths to the .npy image stack, and the CT data, CT block slices and CT counts for calibration.
        op: can be 'train', 'val', or 'test'.
        transform: transforms used for data augmentation
        """
        super().__init__()  # good practice to call the base constructor
        self._op = op
        self._datalist = datalist
        self.norm = norm 
        self._pretrain = pretrain 
        self.n_primal = n_primal
        
    def __len__(self):
        """Database size."""
        return len(self._datalist[1]) #check after constructing datalist 

    def __getitem__(self, index):
        """Stack of nc backprojections for all angles (torch.Size([1, angles*nc, nSliceBlock, shapeCTx, shapeCTz]) ), inaccurate rspCT and accurate rspCT."""
        datalist = self._datalist
        nSliceBlock = datalist[5]
        
        #! analytical CT has different shape 
        path_CT = datalist[1][index]
        sl = datalist[2][index]
        #print(sl)
        path_CTmask = datalist[4][index]
        
        path_CTmask_image = path_CTmask
        
        
        # print(sl,sl+math.floor(nSliceBlock/2)+1,sl-math.floor(nSliceBlock/2))
        mask = np.load(path_CTmask)[:,sl-math.floor(nSliceBlock/2):sl+math.floor(nSliceBlock/2)+1,:]
        mask = mask.transpose(0,2,1)
        CTblock = np.load(path_CT)[:,sl-math.floor(nSliceBlock/2):sl+math.floor(nSliceBlock/2)+1,:]
        CTblock = CTblock.transpose(0,2,1)
        mask_image = np.load(path_CTmask_image)[:,sl-math.floor(nSliceBlock/2):sl+math.floor(nSliceBlock/2)+1,:]
        mask_image = mask_image.transpose(0,2,1)
    
        
        HU_original= np.array([-1400, -1000 ,-800, -600, -400, -200, 0, 200, 400, 600, 800, 1400])
        
        #accurate rspCT
        path_calib = datalist[3][0][index]
        RSP_accurate = np.load(path_calib)
        ctArray = CTblock.flatten('F')
        ictArray = interp1d(HU_original, RSP_accurate, kind= 'linear')(ctArray)
        iCT = np.reshape(ictArray, np.shape(CTblock), order = 'F')
        iCTacc = iCT*mask_image
        
        #system matrix
        Angles = datalist[0]
        pat = path_CT.split('/')[-1].split('_')[2]
        #pat = path_CT.split('/')[-1].split('_')[1]
        #path_local =  r'/project/med2/Ines.Butz/Data/ML/PrimalDual'
        #path = path_local + r'/system_matrices/analytical/'+pat+'_1mm1mm3mm_20add_nions20'
        path = r'data/system_matrices/analytical/'+pat+'_1mm1mm3mm_20add_nions20'
        
        g_angles = []
        sysm_angles = []
        sysm_angles_norm = []
        for i in np.arange(len(Angles)):
            print(path+'/sysm_slice'+str(sl+1)+'_angle'+str(int(Angles[i]))+'.pt')
            g_angles += [np.load(path+r'/proj_slice'+str(sl+1)+'_angle'+str(int(Angles[i]))+'.npy')]
            sysm_angles += [torch.load(path+'/sysm_slice'+str(sl+1)+'_angle'+str(int(Angles[i]))+'.pt')]
            sysm_angles_norm += [torch.load(path+'/sysm_norm_slice'+str(sl+1)+'_angle'+str(int(Angles[i]))+'.pt')]
            
        sysm_angles = torch.stack(sysm_angles)
        sysm_angles_norm = torch.stack(sysm_angles_norm)
        g_angles = np.array(g_angles)
        g_angles = g_angles.transpose(2,0,1)
        
            
        
        #inaccurate rspCT
        path_calib = datalist[3][1][index]
        RSP_inaccurate = np.load(path_calib)
        ctArray = CTblock.flatten('F')
        ictArray = interp1d(HU_original, RSP_inaccurate, kind= 'linear')(ctArray)
        iCT = np.reshape(ictArray, np.shape(CTblock), order = 'F')
        iCTinacc = iCT*mask_image
        
        if self.norm: #pairwise normalisation: divide by maximum RSP value in [iCTacc, iCTinacc] --> caution: divide backprojections by same value!  
            norm_val = np.max([iCTacc, iCTinacc])
        else: 
            norm_val = 1
            
        h0 = np.zeros(g_angles.shape)
        dual = torch.as_tensor(h0.copy(), dtype = torch.float32)
        primal = torch.as_tensor(iCTinacc.copy()/norm_val, dtype = torch.float32)
        
        g = torch.as_tensor(g_angles.copy()/norm_val, dtype = torch.float32)
                
        gt = torch.as_tensor(iCTacc.copy()/norm_val, dtype = torch.float32)
        
        
        mask = torch.as_tensor(mask.copy(), dtype = torch.bool)
        
        
        mask_image = torch.as_tensor(mask_image.copy(), dtype = torch.bool)
        
        
        primal = primal[None,:,:,:].repeat(self.n_primal,1,1,1)
        gt = gt[None,:,:,:]
        mask = mask[None,:,:,:]
        mask_image = mask_image[None,:,:,:]
        #print('here')
        return dual, primal, g, gt, mask, mask_image, sysm_angles, sysm_angles_norm

        
def datalist(pathsCT, paths_CT_calibs, paths_CTmask, train_slices, nangles, nSliceBlock):
    angles = np.linspace(0,180, nangles, endpoint = False)
    CT_slices = []
    calibs_slices_acc = []
    calibs_slices_inacc = []
    paths_CT_all = []
    paths_CT_mask_all = []
    #iterate over all patients (len(paths_stack))
    for p in np.arange(len(pathsCT)):
        
        path_CT = pathsCT[p]
        path_CT_calibs = paths_CT_calibs[p]
        path_CTmask = paths_CTmask[p]
        
        slices = train_slices[p]
        for s in slices:
            calibs_slices_acc += [path_CT_calibs[0] + r'/RSP_accurate_slice'+str(s)+'.npy']
            calibs_slices_inacc += [path_CT_calibs[1] + r'/RSP_inaccurate_slice'+str(s)+'.npy']
            CT_slices += [s]
            paths_CT_all += [path_CT]
            paths_CT_mask_all += [path_CTmask]
   
    return [angles, paths_CT_all, CT_slices, [calibs_slices_acc, calibs_slices_inacc], paths_CT_mask_all, nSliceBlock]
    