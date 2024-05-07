#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:17:39 2023

@author: ines.butz
"""
import numpy as np
import torch
from datapipe.DatasetLPD_LRZ import DatasetPrimalDual, datalist
from networks.LPD_LRZ import primal_dual_ion
#from LPD_OpCorr import primal_dual_ion_modelcorrected
from networks.custom_losses import mask_loss2D
import os

# parser = argparse.ArgumentParser()

# parser.add_argument('--cf', type = int, required=True)
# args = parser.parse_args()

# continue_from = args.cf 
#path to checkpoints
path = 'models/male1_female1_male2_female2_male3_female3_male4_female4_male5_analytical_bs2_primal_dual_ion_20add_nangles2_wopillow_wocp_nions20_checkpoints/'#caution! manually set this path
if os.path.isdir(path) and len(os.listdir(path))>0:
    folders = [x[2] for x in os.walk(path)][0]
    #print(folders)
    epochs = [int(x.split('_')[1].split('.')[0]) for x in folders]
    cof = max(epochs)
    #print(cf)
else:
    cof = -1
# cof = -1


continue_from = cof
#from tqdm.auto import tqdm

# User selected paramters
n_data = 5
n_iter = 10
n_primal = 5
n_dual = 5
nSliceBlock = 7


MagnDeviation = 0.2
Error = 'add'
num_epochs = 2000
save_every = 10
bs = 2
nangles = 2
pretrain_now = 0
norm = 1

pretrained_model = None
Scenario = 'analytical'
mode = 'analytical'
cp = False
#pretrained_model = r'/project/med2/Ines.Butz/Codes/ML/PrimalDual/models/female_male_female2_male2_transmission_primal_dual_nonmasked_20add_nangles79_wopillow_pretrain_checkpoints/cp_759.pt'
#training
if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"

add = '_nions20'#_wopretr'

paths_CT = []
#paths_count_caliborig = []
paths_CT_calibs = []
paths_CTmask = []
train_slices = []
val_slices = []
patients = ['male1', 'female1','male2', 'female2', 'male3', 'female3', 'male4', 'female4', 'male5']

for patient in patients:
    if mode == 'analytical': #paths not adapted for cluster yet (data not uploaded)
        
        nSliceBlock = 7
        paths_CT += [r'data/ReferenceCTs/20230914_analytical_'+patient+'_1mm3mm1mm.npy']
        paths_CTmask += [r'data/ReferenceCTs/20230914_analytical_'+patient+'_1mm3mm1mm_mask.npy']
        # if patient == 'male5':
        #     tr = np.load(r'data/20231011_analytical_'+patient+'_1mm3mm1mm_train_slices.npy')
        #     tr_list = list(tr)
        #     tr_list.remove(10)
        #     train_slices += [np.array(tr_list)]
            
        # else:
        train_slices += [np.load(r'data/20231011_analytical_'+patient+'_1mm3mm1mm_train_slices.npy')]
        val_slices += [np.load(r'data/20231011_analytical_'+patient+'_1mm3mm1mm_val_slices.npy')]
    
        paths_CT_calibs += [[r'data/calibs/analytical_'+patient+'_calibs_acc_'+str(int(100*MagnDeviation))+Error,
                              r'data/calibs/analytical_'+patient+'_calibs_inacc_'+str(int(100*MagnDeviation))+Error]]
        


datalist_train = datalist(paths_CT, paths_CT_calibs, paths_CTmask, train_slices, nangles, nSliceBlock)
datalist_val = datalist(paths_CT, paths_CT_calibs, paths_CTmask, val_slices, nangles, nSliceBlock )


loader_train = torch.utils.data.DataLoader(dataset=DatasetPrimalDual(datalist_train,n_primal,  norm = norm, op = 'train',pretrain = pretrain_now), batch_size=bs, shuffle=True)
loader_val = torch.utils.data.DataLoader(dataset=DatasetPrimalDual(datalist_val, n_primal, norm = norm, op = 'val',pretrain = pretrain_now), batch_size=bs)

model = primal_dual_ion(n_iter=10, n_primal=5, n_ions=20, checkpointing = cp)
name = '_'.join(patients)+'_'+Scenario+'_bs'+str(bs)+'_'+model.__class__.__name__+'_'+str(int(100*MagnDeviation))+Error+'_'+'nangles'+str(len(datalist_train[0]))+'_wopillow'#
if cp == 0:
    name += '_wocp'
if pretrain_now:
    name += '_pretrain'
if pretrained_model != None:
    name += '_pretrained'
name += add
# sysm_angles = []
# for i in np.arange(len(datalist_train[0])):
#     path = r'/project/med2/Ines.Butz/Data/ML/PrimalDual/system_matrices/'+str(nangles)+'Angles/'
#     with open(path+r'sysm_angle'+str(i)+'.pkl', 'rb') as f:
#         sys_coo= pickle.load(f)
#     sysm_angles += [sys_coo]
device = torch.device(dev)

PATH = "models"    
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
    checkpoint = torch.load(path_cp)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_start = checkpoint['epoch']
    losses_train = checkpoint['loss_train']
    losses_val = checkpoint['loss_val']

for epoch in np.arange(epoch_start+1, epoch_start+1+num_epochs):
    print(PATH+r'/'+name+'_checkpoints'+r'/cp_'+str(epoch)+'.pt')
    model.train()  # switch to training mode

    loss_train = 0
    counter= 0
    for dual, primal, g, gt, mask, mask_image, sysm_angles, sysm_angles_norm in loader_train:
        # print(dual.shape, 'dual')
        # print(primal.shape, 'primal')
        # print(g.shape, 'g')
        # print(gt.shape, 'gt')
        # print(mask.shape, 'mask')
        
        optimizer.zero_grad()  # zero optimizer gradients, default option accummulates!
        #outputs = model(dual.to(device), primal.to(device), g.to(device), mask_image, sysm_angles, device, norm,return_all = 0)  # run the forward model
        outputs = model(dual.to(device), primal.to(device), g.to(device),  sysm_angles, sysm_angles_norm, device,return_all = 0)  # run the forward model
        loss = cost(outputs, gt.to(device), mask.to(device))
        
        loss.backward()  # backpropagate loss
        optimizer.step()  # step the optimizer along the gradient

        # keep track of training loss
        loss_train += loss.item()
        print(torch.cuda.max_memory_allocated(device=device))
        torch.cuda.reset_peak_memory_stats(device = device)
    #loss_train /= loader_train.batch_size  # optional, normalize loss by batch size
    losses_train.append(loss_train)
    
    
    # compute and keep track of validation loss
    with torch.no_grad():
        loss_val = 0
        for dual, primal, g, gt, mask, mask_image, sysm_angles, sysm_angles_norm in loader_val:
            #outputs = model(dual.to(device), primal.to(device), g.to(device),mask,  sysm_angles,device,norm, return_all = 0)
            outputs = model(dual.to(device), primal.to(device), g.to(device),  sysm_angles, sysm_angles_norm, device,return_all = 0)
            loss = cost(outputs, gt.to(device), mask.to(device))
            
            loss_val += loss.item()
        loss_val /= loader_val.batch_size
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