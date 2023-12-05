#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:17:39 2023

@author: ines.butz
"""
import numpy as np 
import torch
import pickle
import os
from DatasetLPD import datalist, DatasetPrimalDual
from LPD import primal_dual_straight
from LPD_OpCorr import primal_dual_straight_modelcorrected
from custom_losses import mask_loss2D
from tqdm.auto import tqdm
import time
# User selected paramters
n_data = 5
n_iter = 10
n_primal = 5
n_dual = 5
nSliceBlock = 1


Scenario = 'transmission'
MagnDeviation = 0.05
Error = 'mixed'
num_epochs = 3000
save_every = 20
bs = 16
nangles = 2
norm = 1
pretrain_now = 1
continue_from = 5999#2819#4719
pretrained_model = None
cp = False
#pretrained_model = r'/project/med2/Ines.Butz/Codes/ML/PrimalDual/models/female_male_female2_male2_transmission_primal_dual_nonmasked_20add_nangles79_wopillow_pretrain_checkpoints/cp_759.pt'

#training
if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"

add = ''#'_wopretr'

paths_CT = []
#paths_count_caliborig = []
paths_CT_calibs = []
paths_CTmask = []
train_slices = []
val_slices = []
if pretrain_now:
    patients = ['male1', 'female1', 'male2','female2','male3', 'female3', 'male4','female4', 'male5']
else:
    patients = [ 'MV679', 'MO098', 'HB341','VL053', 'SA659', 'PW341', 'HA229', 'GA795', 'FK619','WA653']


for patient in patients:
    nSliceBlock=1
    if pretrain_now:
        paths_CT += [r'/project/med2/Ines.Butz/Data/ML/PrimalDual/ReferenceCTs/20230914_analytical_'+patient+'_1mm3mm1mm.npy']
        #paths_count_caliborig += [r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/'+patient+'_counts_caliborig.npy']
        paths_CT_calibs += [[r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/calibs/pretraining_'+patient+'_calibs_acc_'+str(int(100*MagnDeviation))+Error,
                              r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/calibs/pretraining_'+patient+'_calibs_inacc_'+str(int(100*MagnDeviation))+Error]]
        paths_CTmask += [r'/project/med2/Ines.Butz/Data/ML/PrimalDual/ReferenceCTs/20230914_analytical_'+patient+'_1mm3mm1mm_mask.npy']
        train_slices += [np.load(r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/20231006_'+patient+'_1mm3mm1mm_train_slices.npy')]
        val_slices += [np.load(r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/20231006_'+patient+'_1mm3mm1mm_val_slices.npy')]
        
    # else:
    #     paths_CT += [r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/'+patient+'_referenceCT.npy']
    #     #paths_count_caliborig += [r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/'+patient+'_counts_caliborig.npy']
    #     paths_CT_calibs += [[r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/calibs/'+patient+'_calibs_acc_'+str(int(100*MagnDeviation))+Error,
    #                           r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/calibs/'+patient+'_calibs_inacc_'+str(int(100*MagnDeviation))+Error]]
    #     paths_CTmask += [r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/'+patient+'_CTmask_wopillow.npy']
    #     train_slices += [np.load(r'/project/med2/Ines.Butz/Codes/ML/DeepBackProj/'+patient+'_train_slices_han_pat.npy')]
    #     val_slices += [np.load(r'/project/med2/Ines.Butz/Codes/ML/DeepBackProj/'+patient+'_val_slices_han_pat.npy')]
        
    

datalist_train = datalist(paths_CT, paths_CT_calibs, paths_CTmask, train_slices, nangles, nSliceBlock)
datalist_val = datalist(paths_CT, paths_CT_calibs, paths_CTmask, val_slices, nangles, nSliceBlock )


loader_train = torch.utils.data.DataLoader(dataset=DatasetPrimalDual(datalist_train, n_primal, norm, op = 'train', pretrain = pretrain_now), batch_size=bs, shuffle=True)
loader_val = torch.utils.data.DataLoader(dataset=DatasetPrimalDual(datalist_val, n_primal, norm, op = 'val', pretrain = pretrain_now), batch_size=bs)
# def reshape_fortran(x, shape):
#     if len(x.shape) > 0:
#         x = x.permute(*reversed(range(len(x.shape))))
#     return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

# data = DatasetPrimalDual(datalist_train, n_primal, norm, op = 'train', pretrain = pretrain_now).__getitem__(30)
# dual, primal, g, gt, mask, mask_image, sysm_angles, sysm_angles_norm = data
# print(primal.shape)
# print(gt.repeat(6,1,1,1).shape, sysm_angles_norm.shape, g.shape )
# gt = gt.repeat(6,1,1,1)
# gt_flat =reshape_fortran(gt, (gt.shape[0],gt.shape[-3]*gt.shape[-2]*gt.shape[-1],1))
# print(gt_flat.shape)
# prod = torch.bmm(sysm_angles_norm, gt_flat)
# a = prod[:,:,0]-g[0,:,:]
# print(torch.max(a[a!=0]))
    
model = primal_dual_straight(n_iter=10, n_primal=5, n_ions=1, checkpointing=cp)

# a,b = np.load(r'/project/med2/Ines.Butz/Data/ML/PrimalDual/nAngles'+str(nangles)+'_opNorms.npy')
# model = primal_dual_straight_modelcorrected(n_iter = 10, sigma = 1/(10*a), tau = 1/(10*b), weightsharing = 1)

name = '_'.join(patients)+'_'+Scenario+'_bs'+str(bs)+'_'+model.__class__.__name__+'_'+str(int(100*MagnDeviation))+Error+'_'+'nangles'+str(len(datalist_train[0]))+'_wopillow'#
if cp==0:
    name += '_wocp'
if pretrain_now:
    name += '_pretrain'
name += add


device = torch.device(dev)

PATH = "/project/med2/Ines.Butz/Codes/ML/PrimalDual/models"    
if not os.path.exists(PATH+r'/'+name+'_checkpoints'):
    os.mkdir(PATH+r'/'+name+'_checkpoints')
learning_rate=1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
cost = mask_loss2D 
if pretrained_model != None and continue_from == -1:
    checkpoint = torch.load(pretrained_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    losses_train = []
    losses_val = []
    epoch_start = -1
    
elif continue_from == -1:    
    model = model.to(device)
    losses_train = []
    losses_val = []
    epoch_start = -1

else:
    path_cp = PATH+r'/'+name+'_checkpoints'+r'/cp_'+str(continue_from)+'.pt'
    print('continuing from', path_cp)
    checkpoint = torch.load(path_cp, map_location = 'cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_start = checkpoint['epoch']
    losses_train = checkpoint['loss_train']
    losses_val = checkpoint['loss_val']

for epoch in tqdm(np.arange(epoch_start+1, epoch_start+1+num_epochs)):
    print(PATH+r'/'+name+'_checkpoints'+r'/cp_'+str(epoch)+'.pt')
    model.train()  # switch to training mode

    loss_train = 0
    for data in loader_train:
        optimizer.zero_grad()  # zero optimizer gradients, default option accummulates!
        
            
       
        dual, primal, g, gt, mask, mask_image, sysm_angles, sysm_angles_norm = data
        #print('loaded batch')
        outputs = model(dual.to(device), primal.to(device), g.to(device),  sysm_angles, sysm_angles_norm, device,return_all = 0)  # run the forward model
        #print('2-after forward pass', torch.cuda.memory_allocated(device)/ (1024 ** 2))
        
        loss = cost(outputs, gt.to(device), mask.to(device))
        
        loss.backward()  # backpropagate loss
        optimizer.step()  # step the optimizer along the gradient

        # keep track of training loss
        loss_train += loss.item()
        
    #loss_train /= loader_train.batch_size  # optional, normalize loss by batch size
    losses_train.append(loss_train)
    
    
    # compute and keep track of validation loss
    with torch.no_grad():
        loss_val = 0
        for data in loader_val:
            dual, primal, g, gt, mask, mask_image, sysm_angles, sysm_angles_norm = data
            outputs = model(dual.to(device), primal.to(device), g.to(device),  sysm_angles, sysm_angles_norm, device,return_all = 0)  # run the forward model
            
            loss = cost(outputs, gt.to(device), mask.to(device))
            
            loss_val += loss.item()
        #loss_val /= loader_val.batch_size
        losses_val.append(loss_val) 
    
    if (epoch+1)%save_every == 0:
        
        path_cp = PATH+r'/'+name+'_checkpoints'+r'/cp_'+str(epoch)+'.pt'
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_train': losses_train,
                    'loss_val': losses_val
                    }, path_cp)
        print('saved checkpoint at'+path_cp)