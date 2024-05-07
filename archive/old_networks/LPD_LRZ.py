#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 12:29:04 2023

@author: ines.butz
"""
import numpy as np
import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint
import time

#3D version for scattered trajectories 
def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

class ConcatenateLayer(nn.Module):
    def __init__(self):
        super(ConcatenateLayer, self).__init__()
    
    def forward(self, *x):
        return torch.cat(list(x), dim=1)

class DualNet(nn.Module):
    def __init__(self,  n_ions):
        super(DualNet, self).__init__()
        
        self.n_channels = 7#3*n_ions
        self.input_concat_layer = ConcatenateLayer()
        layers = [
            nn.Conv2d(self.n_channels, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 5, kernel_size=3, padding=1), #n_ions
        ]
        self.block = nn.Sequential(*layers)
        
    def forward(self, h, Op_f, g):
        x = self.input_concat_layer(h, Op_f, g)
        x = h + self.block(x)
        return x
    
class PrimalNet3D(nn.Module):
    def __init__(self, n_primal):
        super(PrimalNet3D, self).__init__()
        
        self.n_primal = n_primal
        self.n_channels = n_primal + 1
        self.input_concat_layer = ConcatenateLayer()
        layers = [
            nn.Conv3d(self.n_channels, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv3d(32, self.n_primal, kernel_size=3, padding=1),
        ]
        self.block = nn.Sequential(*layers)
        
    def forward(self, f, OpAdj_h):
        x = self.input_concat_layer(f, OpAdj_h)
        x = f + self.block(x)
        return x
        
class primal_dual_ion(nn.Module):
    
    def __init__(self,
                
                primal_architecture = PrimalNet3D,
                dual_architecture = DualNet,
                n_iter = 10,
                n_primal = 5,
                n_ions = 100,
                checkpointing = False):
        
        super(primal_dual_ion, self).__init__()
        self.primal_architecture = primal_architecture
        self.dual_architecture = dual_architecture
        self.n_iter = n_iter
        self.n_primal = n_primal
        self.checkpointing = checkpointing
        
        self.primal_nets = nn.ModuleList()
        self.dual_nets = nn.ModuleList()
        
        for i in range(n_iter):
            self.primal_nets.append(
                primal_architecture(n_primal)
            )
            self.dual_nets.append(
                dual_architecture(n_ions)
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
            # evalop1 = torch.empty(g.shape, dtype = g.dtype).to(device)
            # for j in np.arange(dual.shape[2]):
            #     sys_batches = system_tensors_norm.transpose(0,1)[j].to(device)
            #     img_batches = reshape_fortran(primal[:,1,:,:,:], (primal.shape[0],primal.shape[-3]*primal.shape[-2]*primal.shape[-1],1))
            #     prod = torch.bmm(sys_batches, img_batches)[:,:,0]
            #     prod = prod.reshape(prod.shape[0], primal.shape[-3], int(prod.shape[1]/primal.shape[-3])).transpose(2,1)
            #     print(prod.dtype, evalop1.dtype)
            #     evalop1[:,:,j,:] = prod.detach().clone()
            evalop1 = torch.empty(g.shape).to(device)
            
            for b in np.arange(dual.shape[0]):
                for j in np.arange(dual.shape[2]): #iterate through angles
                    sys_tensor = system_tensors_norm[b][j].to(device)
                    prod = torch.mm(sys_tensor,reshape_fortran(primal[b,1,:,:,:], (primal.shape[-3]*primal.shape[-2]*primal.shape[-1],1)))
                    #print(prod.shape, 'here')
                    #prod = prod.reshape(primal.shape[-3], int(prod.shape[0]/primal.shape[-3])).transpose(1,0)
                    # evalop1[b,:,j,:] = prod
                    #print(evalop1.shape, prod.shape)
                    evalop1[b,0,j,:] = prod[:,0]
                
                       
            if self.checkpointing: 
                dual = checkpoint(self.dual_forward, dual, evalop1,g, k, self.dummy_tensor)
            else: 
                dual = self.dual_forward(dual, evalop1,g, k, self.dummy_tensor)
            if return_all:
                forward_all[k,:,:,:,:] = dual
            
            # evalop2 = torch.empty((dual.shape[2],primal.shape[0],1,primal.shape[2], primal.shape[3], primal.shape[4])).to(device)
            # for j in np.arange(dual.shape[2]):
            #     sys_batch = system_tensors.transpose(0,1)[j].to(device)
            #     dual2 = dual.transpose(1,2).transpose(2,3)
            #     dual2 = dual2.reshape(dual2.shape[0], dual2.shape[1], dual2.shape[2]*dual2.shape[3])
            #     prod = reshape_fortran(torch.bmm(sys_batch,dual2[:,j,:, None]),(primal.shape[0],primal.shape[2],primal.shape[3], primal.shape[4]))
            #     evalop2[j,:,0,:,:,:] = prod.detach().clone()
            evalop2 = torch.empty((dual.shape[2],primal.shape[0],1,primal.shape[2], primal.shape[3], primal.shape[4])).to(device)
            print(dual.shape, 'HERE')
            for b in np.arange(dual.shape[0]):
                for j in np.arange(dual.shape[2]):
                    #dual2 = dual.transpose(1,2).transpose(2,3)
                    #dual2 = dual2.reshape(dual2.shape[0], dual2.shape[1], dual2.shape[2]*dual2.shape[3])
                    dual2 = dual
                    sys_tensor = system_tensors[b][j].to(device)
                    evalop2[j,b,0,:,:,:] = reshape_fortran(torch.mm(sys_tensor,dual2[b,j,:, None]),(primal.shape[2],primal.shape[3], primal.shape[4]))#*1./dual.shape[2]
                
            evalop2 = torch.sum(evalop2, dim = 0)
            
            if self.checkpointing: 
                primal = checkpoint(self.primal_forward, primal, evalop2, k, self.dummy_tensor)
            else: 
                primal = self.primal_forward(primal, evalop2, k, self.dummy_tensor)
            if return_all:
                backward_all[k,:,:,:,:] = primal
            
        if return_all:
            return primal[:,0:1,:,:], forward_all, backward_all
        else:
            return primal[:,0:1,:,:] 
    
    def primal_forward(self, primal, evalop2, k, dummy_tensor):
        return self.primal_nets[k](primal, evalop2)
    
    def dual_forward(self, dual, evalop1, g, k, dummy_tensor):
        return self.dual_nets[k](dual, evalop1, g)