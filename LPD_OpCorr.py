#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:34:04 2023

@author: jenskammerer
"""
import torch.nn as nn
import torch 
from torch.utils.checkpoint import checkpoint
import numpy as np

def double_conv2D(in_channels, out_channels):
    return nn.Sequential(
       nn.Conv2d(in_channels, out_channels, 3, padding=1),
       nn.BatchNorm2d(out_channels),       
       nn.ReLU(inplace=True),
       nn.Conv2d(out_channels, out_channels, 3, padding=1),
       nn.BatchNorm2d(out_channels),
       nn.ReLU(inplace=True))

class UNet2D(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
               
        self.dconv_down1 = double_conv2D(n_in, 64)
        self.dconv_down2 = double_conv2D(64, 128)
        self.dconv_down3 = double_conv2D(128, 256)

       
        self.maxpool = nn.MaxPool2d((1,2))
        #self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.xUp2  = nn.ConvTranspose2d(256,128,(1,2),stride=(1,2),padding=0)
        self.xUp1  = nn.ConvTranspose2d(128,64,(1,2),stride=(1,2),padding=0)
        

        self.dconv_up2 = double_conv2D(128 + 128, 128)
        self.dconv_up1 = double_conv2D(64 + 64, 64)
        self.conv_last = nn.Conv2d(64, n_out, 1)
        
        self.stepsize = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, x):
        inp = x
        #print(x.shape)
        conv1 = self.dconv_down1(x)
        #print(conv1.shape)
        x = self.maxpool(conv1)
        #print(x.shape)
        conv2 = self.dconv_down2(x)
        #print(conv2.shape)
        x = self.maxpool(conv2)
        #print(x.shape)
        conv3 = self.dconv_down3(x)
        
        
        x = self.xUp2(conv3)  
        #print(x.shape, conv2.shape)              
        x = torch.cat([x, conv2], dim=1)      
        x = self.dconv_up2(x)
        x = self.xUp1(x)        
        
        x = torch.cat([x, conv1], dim=1)         
        x = self.dconv_up1(x)
        update = self.conv_last(x)

        return inp + self.stepsize * update

def double_conv3D(in_channels, out_channels):
    return nn.Sequential(
       nn.Conv3d(in_channels, out_channels, 3, padding=1),
       nn.BatchNorm3d(out_channels),       
       nn.ReLU(inplace=True),
       nn.Conv3d(out_channels, out_channels, 3, padding=1),
       nn.BatchNorm3d(out_channels),
       nn.ReLU(inplace=True))

class UNet3D(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
               
        self.dconv_down1 = double_conv3D(n_in, 64)
        self.dconv_down2 = double_conv3D(64, 128)
        self.dconv_down3 = double_conv3D(128, 256)

       
        self.maxpool = nn.MaxPool3d((2,2,1))
        #self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.xUp2  = nn.ConvTranspose3d(265,128,(2,2,1),stride=(2,2,1),padding=0)
        self.xUp1  = nn.ConvTranspose3d(128,64,(2,2,1),stride=(2,2,1),padding=0)
        

        self.dconv_up2 = double_conv3D(128 + 128, 128)
        self.dconv_up1 = double_conv3D(64 + 64, 64)
        self.conv_last = nn.Conv3d(64, n_out, 1)
        
        self.stepsize = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, x):
        inp = x
        
        conv1 = self.dconv_down1(x)
        print(conv1.shape)
        x = self.maxpool(conv1)
        print(x.shape)
        conv2 = self.dconv_down2(x)
        print(conv2.shape)
        x = self.maxpool(conv2)
        print(x.shape)
        conv3 = self.dconv_down3(x)
        
        
        
        x = self.xUp2(conv3)                
        x = torch.cat([x, conv2], dim=1)      
        x = self.dconv_up2(x)
        x = self.xUp1(x)        
        
        x = torch.cat([x, conv1], dim=1)         
        x = self.dconv_up1(x)
        update = self.conv_last(x)

        return inp + self.stepsize * update

def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


   
class primal_dual_straight_modelcorrected(nn.Module):
    
    def __init__(self,
                
                primal_architecture = UNet2D,
                dual_architecture = UNet2D,
                n_iter = 10,
                sigma = 1e-4,#check
                tau = 1e-4, #check 
                weightsharing = 1):
        
        super(primal_dual_straight_modelcorrected, self).__init__()
        self.primal_architecture = primal_architecture
        self.dual_architecture = dual_architecture
        self.n_iter = n_iter
        self.sigma = sigma
        self.tau = tau
        self.ws = weightsharing
        
        self.primal_nets = nn.ModuleList()
        self.dual_nets = nn.ModuleList()
        
        if self.ws:
            for i in range(n_iter):
                self.primal_nets.append(
                    primal_architecture(5,5)
                )
                self.dual_nets.append(
                    dual_architecture(1,1)
                )
            
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
            
    def forward(self, dual,primal,g, sysm_angles, sysm_angles_norm, device,return_all):
        
        #primal.requires_grad=True ##warum
        if return_all:
            forward_all = torch.empty([self.n_iter]+list(dual.shape))
            backward_all = torch.empty([self.n_iter]+list(primal.shape))
        
        system_tensors = sysm_angles
        system_tensors_norm = sysm_angles_norm #normalisation of system matrix (to avoid vanishing (both forward and back normalised) or explosion (both not normalised) of image values) in forward projection
        
        
        for k in np.arange(self.n_iter):
            evalop1 = torch.empty(g.shape).to(device)
            for j in np.arange(dual.shape[2]):
                sys_batches = system_tensors_norm.transpose(0,1)[j].to(device)
                img_batches = reshape_fortran(primal[:,1,:,:,:], (primal.shape[0],primal.shape[-3]*primal.shape[-2]*primal.shape[-1],1))
                prod = torch.bmm(sys_batches, img_batches)[:,:,0]
                prod = prod.reshape(prod.shape[0], primal.shape[-3], int(prod.shape[1]/primal.shape[-3])).transpose(2,1)
                evalop1[:,:,j,:] = prod
                
                       
            dual = checkpoint(self.dual_forward, dual, evalop1, g, k,self.sigma, self.dummy_tensor)
            if return_all:
                forward_all[k,:,:,:,:] = dual
            
            evalop2 = torch.empty((dual.shape[2],primal.shape[0],1,primal.shape[2], primal.shape[3], primal.shape[4])).to(device)
            for j in np.arange(dual.shape[2]):
                sys_batch = system_tensors.transpose(0,1)[j].to(device)
                dual2 = dual.transpose(1,2).transpose(2,3)
                dual2 = dual2.reshape(dual2.shape[0], dual2.shape[1], dual2.shape[2]*dual2.shape[3])
                prod = reshape_fortran(torch.bmm(sys_batch,dual2[:,j,:, None]),(primal.shape[0],primal.shape[2],primal.shape[3], primal.shape[4]))
                evalop2[j,:,0,:,:,:] = prod
            evalop2 = torch.sum(evalop2, dim = 0)
            
            primal = checkpoint(self.primal_forward, primal.squeeze(dim=4), evalop2.squeeze(dim=4), k, self.tau, self.dummy_tensor)
            primal = primal.unsqueeze(dim=4)
            if return_all:
                backward_all[k,:,:,:,:] = primal
            
        if return_all:
            return primal[:,0:1,:,:], forward_all, backward_all
        else:
            return primal[:,0:1,:,:] 
    
    def primal_forward(self, primal, evalop2, k, tau,dummy_tensor):
        if self.ws:
            G_theta = self.primal_nets[k](primal - tau*evalop2)
        else: 
            G_theta = self.primal_architecture(primal -tau*evalop2)
        return G_theta
    
    def dual_forward(self, dual, evalop1, g, k, sigma, dummy_tensor):
        if self.ws:
            f_phi = self.dual_nets[k](evalop1)
        else: 
            f_phi = self.dual_architecture(evalop1)
        return dual+ sigma*(f_phi-g)