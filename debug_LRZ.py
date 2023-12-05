#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 12:30:31 2023

@author: ines.butz
"""

import numpy as np
import torch
from DatasetLPD_LRZ import DatasetPrimalDual, datalist
from LPD_LRZ_revert import primal_dual_ion
#from LPD_OpCorr import primal_dual_ion_modelcorrected
from custom_losses import mask_loss2D
import time
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#Test
nangles = 2
n_primal = 5

MagnDeviation = 0.2
Error = 'add'
pretrain_now = 0
down = 1
norm = 1
bs = 1
run_local= 1
paths_CT = []
#paths_count_caliborig = []
paths_CT_calibs = []
paths_CTmask = []
train_slices = []
val_slices = []
# if down:
#     patients = ['female1', 'male1', 'male2']
# else:
#     patients = ['male']
patients = ['female1']
if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"

device = torch.device(dev)
patients = ['male1', 'female1', 'male4', 'female4']
if run_local: 
    path = r'/project/med2/Ines.Butz/Data/ML/PrimalDual'
else:
    path = 'data'
for patient in patients:
    
    nSliceBlock = 7
    paths_CT += [path+r'/ReferenceCTs/20230914_analytical_'+patient+'_1mm3mm1mm.npy']
    paths_CTmask += [path+r'/ReferenceCTs/20230914_analytical_'+patient+'_1mm3mm1mm_mask.npy']
    train_slices += [np.load(path+r'/20231011_analytical_'+patient+'_1mm3mm1mm_train_slices.npy')]
    val_slices += [np.load(path+r'/20231011_analytical_'+patient+'_1mm3mm1mm_val_slices.npy')]
    

    paths_CT_calibs += [[path+r'/calibs/analytical_'+patient+'_calibs_acc_'+str(int(100*MagnDeviation))+Error,
                          path+r'/calibs/analytical_'+patient+'_calibs_inacc_'+str(int(100*MagnDeviation))+Error]]
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
datalist_train = datalist(paths_CT, paths_CT_calibs, paths_CTmask, train_slices, nangles, nSliceBlock)

loader_train = torch.utils.data.DataLoader(dataset=DatasetPrimalDual(datalist_train,  n_primal, norm = norm, op = 'train',pretrain = pretrain_now), batch_size=bs, shuffle=True)
model = primal_dual_ion(n_iter = 10, n_primal = n_primal, n_ions = 20)#100
#model = primal_dual_ion_modelcorrected(n_iter = 10, sigma = 1e-4, tau = 1e-4, weightsharing = 1)

print(count_parameters(model))
print('0-beginning mem',  torch.cuda.memory_allocated(device)/ (1024 ** 2))
model = model.to(device)
print('1-after model to device', torch.cuda.memory_allocated(device)/ (1024 ** 2))
counter = 0
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
cost = mask_loss2D 
counter = 0
dataload = []
model_t = []
loss_t = []
back_t = []
optim_t = []
total_t = []

   
for data in loader_train:
    start = time.time()
    dual, primal, g, gt, mask, mask_image, sysm_angles, sysm_angles_norm= data
    print(dual.shape, g.shape, sysm_angles.shape, sysm_angles_norm.shape)
    
    load_data = time.time()
    diff0 = load_data-start
    print(diff0,'load_data')
    dataload += [diff0]
        
    #a=  torch.cuda.memory_allocated(device)/ (1024 ** 2)
    outputs = model(dual.to(device), primal.to(device), g.to(device),  sysm_angles, sysm_angles_norm, device,return_all = 0)  # run the forward model
    model_time = time.time()
    diff1 = model_time - load_data
    print(model_time - load_data, 'model')
    model_t += [diff1]
    #b = torch.cuda.memory_allocated(device)/ (1024 ** 2)
    #print('2-after forward pass', torch.cuda.memory_allocated(device)/ (1024 ** 2))
    #print('3-memory consumed by forward pass', b-a)
    loss = cost(outputs, gt.to(device), mask.to(device))
    loss_time = time.time()
    diff2 = loss_time-model_time
    print(loss_time-model_time, 'loss')
    loss_t += [diff2]
    #print('4-memory after loss eval', torch.cuda.memory_allocated(device)/ (1024 ** 2))
    loss.backward()
    back = time.time()
    diff3 = back-loss_time
    print(back-loss_time, 'backprop')
    back_t += [diff3]
    #print('5-memory after backward pass', torch.cuda.memory_allocated(device)/ (1024 ** 2))
    optimizer.step()
    optim = time.time()
    diff4 = optim - back
    print(optim - back, 'optim')
    optim_t += [diff4]
    #print('6-memory after optimizer step', torch.cuda.memory_allocated(device)/ (1024 ** 2))
    end = time.time()
    total= end-start
    print(end-start, 'total')
    total_t += [total]
    counter += 1
    if counter == 10: 
        break
print(np.mean(dataload), 'mean dataload')
print(np.mean(model_t), 'mean model')
print(np.mean(loss_t), 'mean loss')
print(np.mean(back_t), 'mean back')
print(np.mean(optim_t), 'mean optim')
print(np.mean(total_t), 'mean total')
gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
print(gpu_memory)