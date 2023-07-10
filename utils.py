# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 18:33:09 2023

@author: nick8
"""

import os
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms 
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image


"""Functions for saving and restoring checkpoints"""
def restore_checkpoint(ckpt_dir, state, device):
    """Taken from https://github.com/yang-song/score_sde_pytorch"""
    if not os.path.exists(ckpt_dir):
        Path(os.path.dirname(ckpt_dir)).mkdir(parents=True, exist_ok=True)
        print(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['step'] = loaded_state['step']
        return state

def save_checkpoint(ckpt_dir, state):
    """Taken from https://github.com/yang-song/score_sde_pytorch"""

    saved_state = {
            'optimizer': state['optimizer'].state_dict(),
            'model': state['model'].state_dict(),
            'step': state['step']
        }
    torch.save(saved_state, ckpt_dir)

def save_png(save_dir, data, name, nrow=None):
    """Save tensor 'data' as a PNG"""
    if nrow == None:
        nrow = int(np.sqrt(data.shape[0]))
    image_grid = make_grid(data, nrow, padding=2)
    with open(os.path.join(save_dir, name), "wb") as fout:
        save_image(image_grid, fout)


"""Optimizer that depends on steps not epochs"""
def optimization_manager():
   
    def optimize_fn(optimizer, params, step, lr= 2e-4,
                    warmup=5000,
                    grad_clip=1):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()
    return optimize_fn

def count_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pytorch_total_params



def get_model(name = "ResNet"):
    
    if name == "ResNet":
        model = resnet50()
        model.conv1 = torch.nn.Conv2d(1,64, kernel_size=(7,7), stride = (2,2), padding = (3,3), bias = False)
        model = nn.Sequential(model, nn.Linear(1000,10), nn.LogSoftmax(1))
        
    if name == "MLP":
        
        model = nn.Sequential(nn.Flatten(),nn.Linear(28*28,300),nn.Linear(300, 10))
    
    
    return model

class MLP(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_dim,300)
        self.linear2 = nn.Linear(300,output_dim)

    def forward(self,x):
        out = self.flatten(x)
        out = self.linear1(x)
        out = self.linear2(x)

        return out
