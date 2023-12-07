#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:21:57 2023

@author: ines.butz
"""
import numpy as np 
import torch
import pickle
from datapipe.DatasetLPD import datalist, DatasetPrimalDual
from networks.LPD import primal_dual_straight
from networks.custom_losses import mask_loss2D
import matplotlib.pyplot as plt
import math

# User selected parameters
n_data = 5
n_iter = 10
n_primal = 5
n_dual = 5
nSliceBlock = 1

Scenario = 'transmission'
MagnDeviation = 0.05
Error = 'mixed'
nangles = 2
norm = 1
ep = 7279#359#4839#159#119#99#
bs = 16
cp = False
pretrained_model = None
#pretrained_model = r'/project/med2/Ines.Butz/Codes/ML/PrimalDual/models/female_male_female2_male2_transmission_primal_dual_nonmasked_20add_nangles79_wopillow_pretrain_checkpoints/cp_759.pt'

# a,b = np.load(r'/project/med2/Ines.Butz/Data/ML/PrimalDual/nAngles'+str(nangles)+'_opNorms.npy')
# model = primal_dual_straight_modelcorrected(n_iter = 10, sigma = 1/(10*a), tau = 1/(10*b), weightsharing = 1)
model = primal_dual_straight(n_iter=10, n_primal=5, n_ions=1, checkpointing=cp)

cost = mask_loss2D 
return_all = 1
pretrain_now = 1
save = 0

add = ''#'_wopretr'#''#

save_plots = 0
save_path = r'/project/med2/Ines.Butz/Presentations/ProjectUpdates/20231005/'
save_name = r'LPD_pretrain5'

# Testing
if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
if pretrain_now:
    #patients_test = ['female5']
    patients_test =  ['male1', 'female1', 'male2','female2','male3', 'female3', 'male4','female4', 'male5']
    patients = ['male1', 'female1', 'male2','female2','male3', 'female3', 'male4','female4', 'male5']
    
# else: 
#     patients_test = ['MH488']


name = '_'.join(patients)+'_'+Scenario+'_bs'+str(bs)+'_'+model.__class__.__name__+'_'+str(int(100*MagnDeviation))+Error+'_'+'nangles'+str(nangles)+'_wopillow'
# if 'nonmasked' in name:
#     name = name.replace('nonmasked_','')
name += add
if cp==0:
    name += '_wocp'

if pretrain_now: 
    name += '_pretrain'
if pretrained_model != None: 
    name += '_pretrained'
paths_CT = []
#paths_count_caliborig = []
paths_CT_calibs = []
paths_CTmask = []
test_slices = []
for patient in patients_test:

    if pretrain_now:
        paths_CT += [r'/project/med2/Ines.Butz/Data/ML/PrimalDual/ReferenceCTs/20230914_analytical_'+patient+'_1mm3mm1mm.npy']
        paths_CT_calibs += [[r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/calibs/pretraining_'+patient+'_calibs_acc_'+str(int(100*MagnDeviation))+Error,
                              r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/calibs/pretraining_'+patient+'_calibs_inacc_'+str(int(100*MagnDeviation))+Error]]
        paths_CTmask += [r'/project/med2/Ines.Butz/Data/ML/PrimalDual/ReferenceCTs/20230914_analytical_'+patient+'_1mm3mm1mm_mask.npy']
        test_slices += [np.load(r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/20231006_'+patient+'_1mm3mm1mm_val_slices.npy')]
        shape_primal = (0,1,256,256,1)
    # else:
    #     paths_CT += [r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/'+patient+'_referenceCT.npy']
    #     #paths_count_caliborig += [r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/'+patient+'_counts_caliborig.npy']
    #     paths_CT_calibs += [[r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/calibs/'+patient+'_calibs_acc_'+str(int(100*MagnDeviation))+Error,
    #                           r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/calibs/'+patient+'_calibs_inacc_'+str(int(100*MagnDeviation))+Error]]
    #     paths_CTmask += [r'/project/med2/Ines.Butz/Data/ML/DeepBackProj/'+patient+'_CTmask_wopillow.npy']
    #     test_slices += [np.load(r'/project/med2/Ines.Butz/Codes/ML/DeepBackProj/'+patient+'_test_slices_han_pat.npy')]
    #     shape_primal = (0,1,314,314)
datalist_test = datalist(paths_CT, paths_CT_calibs, paths_CTmask, test_slices, nangles, nSliceBlock)
with open(r'/project/med2/Ines.Butz/Codes/ML/PrimalDual/Results/datalist_test'+name+'.pkl', 'wb') as f:
    pickle.dump(datalist_test,f)
loader_test = torch.utils.data.DataLoader(dataset=DatasetPrimalDual(datalist_test, n_primal, norm, op = 'train', pretrain = pretrain_now))

PATH = "/project/med2/Ines.Butz/Codes/ML/PrimalDual/models"
path_cp = PATH+r'/'+name+'_checkpoints'+r'/cp_'+str(ep)+'.pt'

if dev == 'cpu':
    checkpoint = torch.load(path_cp,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'] )
else: 
    checkpoint = torch.load(path_cp)
    model.load_state_dict(checkpoint['model_state_dict'])

losses_train = checkpoint['loss_train']
# print(losses_train)
losses_val = checkpoint['loss_val']
device = torch.device(dev)
model = model.to(device)

#shape_dual = (0,1,nangles,314)
CT_inacc = np.empty(shape_primal)
CT_acc = np.empty(shape_primal)
CT_pred = np.empty(shape_primal)
masks = np.empty(shape_primal)
with torch.no_grad():
    model.eval()
    MSE_ref = []
    MSE_pred = []
    counter = 0
    for data in loader_test:
        
        dual, primal, g, gt, mask, mask_image, sysm_angles, sysm_angles_norm = data
        print(primal.shape,'primal')
        outputs = model(dual.to(device), primal.to(device), g.to(device),  sysm_angles, sysm_angles_norm, device,return_all = 0)
        
        predictions = outputs.cpu()
        MSE_ref += [cost(primal[:,0:1,:,:], gt, mask)]
        MSE_pred += [cost(predictions, gt, mask)]
        
        CT_inacc = np.append(CT_inacc, primal[:,0:1,:,:].cpu().numpy(), axis = 0)
        CT_acc = np.append(CT_acc, gt.cpu().numpy(), axis = 0)
        CT_pred = np.append(CT_pred, predictions.numpy(), axis = 0)
        masks = np.append(masks, mask.cpu().numpy(), axis = 0)
        counter += 1
        print(counter)
        
        # if counter == 25:
        #     fig, axs = plt.subplots(2,5 )
        #     plt.subplots_adjust(hspace = 0)
        #     fig2, axs2 = plt.subplots(2,5)
        #     mins = []
        #     maxs = []
        #     mins2 = []
        #     maxs2 = []
        #     for m in np.arange(10):
                
        #         mins += [(np.abs(bw[m,0,0,:,:]-gt[0,0,:,:])).min()]
        #         maxs += [(np.abs(bw[m,0,0,:,:]-gt[0,0,:,:])).max()]
                
        #         mins2 += [(np.abs(fw[m,0,0,:,:]-g[0,0,:,:])).min()]
        #         maxs2 += [(np.abs(fw[m,0,0,:,:]-g[0,0,:,:])).max()]
        #     gmin1 = np.min(mins)
        #     gmax1 = np.max(maxs)
        #     gmin2 = np.min(mins2)
        #     gmax2 = np.max(maxs2)
        #     for i in np.arange(2):
        #         for j in np.arange(5):
        #             im1 = axs[i,j].imshow(np.abs(bw[i*5+j,0,0,:,:]-gt[0,0,:,:]), cmap = 'gray', vmin = gmin1, vmax = gmax1)
        #             #axs[i,j].set_title(np.abs(bw[i*5+j,0,0,:,:]-gt[0,0,:,:]).mean())
                    
        #             im2 = axs2[i,j].imshow(np.abs(fw[i*5+j,0,0,:,:]-g[0,0,:,:]), cmap = 'gray', vmin = gmin2, vmax = gmax2, aspect = 3)
        #             #axs2[i,j].set_title(np.abs(fw[i*5+j,0,0,:,:]-g[0,0,:,:]).mean())
        #     fig.colorbar(im1, ax=axs.ravel().tolist(), shrink = 0.75)
        #     fig2.colorbar(im2, ax=axs2.ravel().tolist(), shrink = 0.75)
        #     plt.figure()
        #     plt.imshow(gt[0,0,:,:], cmap = 'gray')
        #     plt.imshow(mask.cpu().numpy()[0,0,:,:], alpha = 0.5)
        #     plt.show()
        
            
   
        
predictedCT = np.squeeze(CT_pred, axis = 1)
accCT = np.squeeze(CT_acc, axis = 1)
inaccCT = np.squeeze(CT_inacc, axis = 1)
mask_array = np.squeeze(masks, axis = 1)

fig, ax = plt.subplots(figsize=(4,4), layout = 'compressed')
lines1 = ax.violinplot([MSE_ref, MSE_pred], showmeans = True)
print(np.mean(MSE_ref),'REF')
print(np.mean(MSE_pred),'ML')
ax.set_xticks([1, 2])
ax.set_ylabel('MSE')
ax.set_xticklabels(['$CT_{cal}$', 'LPD'])
if save_plots: 
    plt.savefig(save_path+r'/MSE_'+save_name+'.png')
plt.show()

if pretrain_now:
     indices = [20, 35, 60] #phantom
else:
     indices = [25,43,15]
if len(shape_primal) == 4:
    gmax = np.max([predictedCT[indices,:,:]*mask_array[indices,:,:], accCT[indices,:,:]*mask_array[indices,:,:]])
    gmin = np.min([predictedCT[indices,:,:]*mask_array[indices,:,:], accCT[indices,:,:]*mask_array[indices,:,:]])
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(12,8), layout = 'compressed')
    for i in np.arange(len(indices)):
        im1 = axs[0,i].imshow(predictedCT[indices[i],:,:]*mask_array[indices[i],:,:], cmap = 'gray', vmin = gmin, vmax = gmax)
        axs[0,i].set_title('$CT_{inf}$')
        im2 = axs[1,i].imshow(accCT[indices[i],:,:]*mask_array[indices[i],:,:], cmap = 'gray', vmin = gmin, vmax = gmax)
        axs[1,i].set_title('$CT_{true}$')
    plt.colorbar(im1, ax=axs[0,:])
    plt.colorbar(im2, ax=axs[1,:])
    plt.show()
    
    gmax = np.max([np.abs(predictedCT[indices,:,:]*mask_array[indices,:,:]- accCT[indices,:,:]*mask_array[indices,:,:]),np.abs(inaccCT[indices,:,:]*mask_array[indices,:,:]- accCT[indices,:,:]*mask_array[indices,:,:]) ])
    gmin = np.min([np.abs(predictedCT[indices,:,:]*mask_array[indices,:,:]- accCT[indices,:,:]*mask_array[indices,:,:]),np.abs(inaccCT[indices,:,:]*mask_array[indices,:,:]- accCT[indices,:,:]*mask_array[indices,:,:]) ])
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(12,8), layout = 'compressed')
    for i in np.arange(len(indices)):
        im1 = axs[0,i].imshow(np.abs(predictedCT[indices[i],:,:]*mask_array[indices[i],:,:]- accCT[indices[i],:,:]*mask_array[indices[i],:,:]), cmap = 'gray', vmin = gmin, vmax = gmax)
        axs[0,i].set_title('|$CT_{inf} - CT_{true}$|')
        im2 = axs[1,i].imshow(np.abs(inaccCT[indices[i],:,:]*mask_array[indices[i],:,:]- accCT[indices[i],:,:]*mask_array[indices[i],:,:]), cmap = 'gray', vmin = gmin, vmax = gmax)
        axs[1,i].set_title('|$CT_{cal} - CT_{true}$|')
    plt.colorbar(im1, ax=axs[0,:])
    plt.colorbar(im2, ax=axs[1,:])
    if save_plots: 
        plt.savefig(save_path+r'/diff_'+save_name+'.png')
    plt.show()
if len(shape_primal) == 5:
    mid = math.floor(nSliceBlock/2)
    gmax = np.max([predictedCT[indices,:,:, mid]*mask_array[indices,:,:, mid], accCT[indices,:,:, mid]*mask_array[indices,:,:, mid]])
    gmin = np.min([predictedCT[indices,:,:, mid]*mask_array[indices,:,:, mid], accCT[indices,:,:, mid]*mask_array[indices,:,:, mid]])
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(12,8), layout = 'compressed')
    for i in np.arange(len(indices)):
        im1 = axs[0,i].imshow(predictedCT[indices[i],:,:, mid]*mask_array[indices[i],:,:, mid], cmap = 'gray', vmin = gmin, vmax = gmax)
        axs[0,i].set_title('$CT_{inf}$')
        im2 = axs[1,i].imshow(accCT[indices[i],:,:, mid]*mask_array[indices[i],:,:, mid], cmap = 'gray', vmin = gmin, vmax = gmax)
        axs[1,i].set_title('$CT_{true}$')
    plt.colorbar(im1, ax=axs[0,:])
    plt.colorbar(im2, ax=axs[1,:])
    plt.show()
    
    gmax = np.max([np.abs(predictedCT[indices,:,:, mid]*mask_array[indices,:,:, mid]- accCT[indices,:,:, mid]*mask_array[indices,:,:,mid]),np.abs(inaccCT[indices,:,:, mid]*mask_array[indices,:,:, mid]- accCT[indices,:,:, mid]*mask_array[indices,:,:,mid]) ])
    gmin = np.min([np.abs(predictedCT[indices,:,:, mid]*mask_array[indices,:,:, mid]- accCT[indices,:,:, mid]*mask_array[indices,:,:, mid]),np.abs(inaccCT[indices,:,:,mid]*mask_array[indices,:,:,mid]- accCT[indices,:,:, mid]*mask_array[indices,:,:,mid]) ])
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(12,8), layout = 'compressed')
    for i in np.arange(len(indices)):
        im1 = axs[0,i].imshow(np.abs(predictedCT[indices[i],:,:, mid]*mask_array[indices[i],:,:, mid]- accCT[indices[i],:,:, mid]*mask_array[indices[i],:,:, mid]), cmap = 'gray', vmin = gmin, vmax = gmax)
        axs[0,i].set_title('|$CT_{inf} - CT_{true}$|')
        im2 = axs[1,i].imshow(np.abs(inaccCT[indices[i],:,:, mid]*mask_array[indices[i],:,:, mid]- accCT[indices[i],:,:, mid]*mask_array[indices[i],:,:, mid]), cmap = 'gray', vmin = gmin, vmax = gmax)
        axs[1,i].set_title('|$CT_{cal} - CT_{true}$|')
    plt.colorbar(im1, ax=axs[0,:])
    plt.colorbar(im2, ax=axs[1,:])
    if save_plots: 
        plt.savefig(save_path+r'/diff_'+save_name+'.png')
    plt.show()

fig, ax = plt.subplots(figsize=(8,4), layout = 'compressed')
lines1 = ax.plot(losses_train, 'C0')
ax.set_ylabel('Train Loss')
axr = ax.twinx()  # show validation loss on the right axes
lines2 = axr.plot(losses_val, 'C1')
axr.axhline(min(losses_val), c='C2', ls=':')
axr.axvline(np.argmin(losses_val), c='C2', ls=':')
axr.set_ylabel('Val Loss')
ax.set_xlabel('Epoch')
ax.legend([lines1[0], lines2[0]], ['train', 'val'])
plt.show() 

fig, ax = plt.subplots(figsize=(8,4), layout = 'compressed')
lines1 = ax.plot(losses_train, 'C0')
ax.set_ylabel('Train Loss')
axr = ax.twinx()  # show validation loss on the right axes
lines2 = axr.plot(losses_val, 'C1')
axr.axhline(min(losses_val), c='C2', ls=':')
axr.axvline(np.argmin(losses_val), c='C2', ls=':')
val_ax = axr.get_ylim()
axr.set_ylabel('Val Loss')
ax.set_xlabel('Epoch')
axr.set_ylim(np.min([min(losses_val)*0.98,np.mean(losses_val[-100:])*0.7]), np.mean(losses_val[-100:])*2)
train_ax = ax.get_ylim()
span_train = train_ax[1]-train_ax[0]
span_val = val_ax[1]-val_ax[0]
ratio = span_train/span_val
window_val1 = axr.get_ylim()[0]-val_ax[0]
window_val2 = axr.get_ylim()[1]-val_ax[0]
bottom_train_new = train_ax[0]+window_val1*ratio
top_train_new = train_ax[0]+window_val2*ratio
ax.set_ylim(bottom_train_new, top_train_new)
ax.legend([lines1[0], lines2[0]], ['train', 'val'])
if save_plots: 
    plt.savefig(save_path+r'/loss_'+save_name+'.png')
plt.show() 

if save:
    if ep != -1:
        np.save("/project/med2/Ines.Butz/Codes/ML/PrimalDual/models/"+name+'_'+str(ep)+'CTpred.npy', predictedCT)
        np.save("/project/med2/Ines.Butz/Codes/ML/PrimalDual/models/"+name+'_'+str(ep)+'CTacc.npy', accCT)
        np.save("/project/med2/Ines.Butz/Codes/ML/PrimalDual/models/"+name+'_'+str(ep)+'CTinacc.npy', inaccCT)
        np.save("/project/med2/Ines.Butz/Codes/ML/PrimalDual/models/"+name+'_'+str(ep)+'mask.npy', mask_array)
    else:
        np.save("/project/med2/Ines.Butz/Codes/ML/PrimalDual/models/"+name+'CTpred.npy', predictedCT)
        np.save("/project/med2/Ines.Butz/Codes/ML/PrimalDual/models/"+name+'CTacc.npy', accCT)
        np.save("/project/med2/Ines.Butz/Codes/ML/PrimalDual/models/"+name+'CTinacc.npy', inaccCT)
        np.save("/project/med2/Ines.Butz/Codes/ML/PrimalDual/models/"+name+'mask.npy', mask_array)
    