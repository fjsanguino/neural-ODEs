# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 19:08:04 2023

@author: nick8
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from torchvision import datasets
from torchvision import transforms 
from torch.utils.data import DataLoader
import time
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader, Dataset
from numpy import random
from torch.optim import Adam
from torch import nn
from pathlib import Path
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
from utils import get_model, restore_checkpoint, save_checkpoint, count_params,save_png, optimization_manager

#set random seed for reproducibility

def seed_init_fn(x):
   #seed = args.seed + x
   seed = x
   np.random.seed(seed)
   random.seed(seed)
   torch.manual_seed(seed)
   return
seed_init_fn(0)

"""Set hyper-parameters"""
IMG_SIZE = 28
BATCH_SIZE = 32
num_train_steps = 5000
snapshot_freq_for_preemption = 2000
eval_freq = 1000
snapshot_freq = 5000
sample_freq = 2000
model_name = "ResNet"
device = "cuda" if torch.cuda.is_available() else "cpu"



"""Create/set checpoint and sample directories"""
checkpoint_dir = os.path.join(os.getcwd(), "checkpoint",model_name)
checkpoint_meta_dir = os.path.join(
        os.getcwd(), "checkpoint-meta",model_name, "checkpoint.pth")
sample_dir = os.path.join(os.getcwd(), "samples_dir", model_name)

if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


"""Instantiate dataloaders of training and test datasets"""
transform = [transforms.Resize(IMG_SIZE),
                 transforms.CenterCrop(IMG_SIZE), transforms.ToTensor()]
transform = transforms.Compose(transform)
training_data = datasets.MNIST(
            root=".", train=True, download=True, transform=transform)
test_data = datasets.MNIST(
            root=".", train=False, download=True, transform=transform)


trainloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


"""Define the loss function"""
def get_loss():
    
    def loss_fn(model,batch, labels):
        pred  = model(batch)
        return F.nll_loss(pred, labels)
    
    return loss_fn


    


"""Generate  a sample image and save it"""   
def sample(sample_dir = sample_dir, step = None):
    if step is None:
        step = str(0)
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    sample_img, label = next(iter(testloader))
    pred = model(sample_img)
    for i in range(0,6):
        
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(sample_img[i][0], cmap='gray', interpolation='none')
        plt.title(torch.argmax(pred[i]))
        plt.xticks([])
        plt.yticks([])
    plt.savefig(os.path.join(sample_dir,"sample_step_"+step))
    
"""Wrapper for one training/test step"""
def get_step_fn(train,optimize_fn=optimization_manager(),
                 device=None):
    """A wrapper for loss functions in training or evaluation
    Based on code from https://github.com/yang-song/score_sde_pytorch"""
    if device == None:
        device = "cpu"

    loss_fn = get_loss()


    def step_fn(state, batch, labels):
        """Running one step of training or evaluation.
        Returns:
                loss: The average loss value of this state.
        """
        model = state['model']
        if train:
            optimizer = state['optimizer']
            optimizer.zero_grad()
            loss = loss_fn(model, batch, labels)
            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state['step'])
            state['step'] += 1
            
        else:
            with torch.no_grad():
                loss = loss_fn(model, batch, labels)
                

        return loss

    return step_fn


"""Get model,  input is string with model name"""
model = get_model(model_name)
model.to(device)


print("Total params:",  count_params(model)) 

optimizer = Adam(model.parameters(), lr=0.001)



"""Initialise model state"""

state = dict(optimizer=optimizer, model=model, step=0)
state = restore_checkpoint(checkpoint_meta_dir, state, device)
optimize_fn = optimization_manager()
train_step_fn = get_step_fn(train=True,optimize_fn=optimize_fn)
eval_step_fn = get_step_fn(train=False,optimize_fn=optimize_fn)

"""Initialise training iterables"""
initial_step = int(state['step'])
train_iter = iter(trainloader)
eval_iter = iter(testloader)


"""Training Loop"""
train = False

if (train == True):

    for step in range(initial_step, num_train_steps + 1):
        # Train step
        try:
            batch, labels = next(train_iter)
            batch = batch.to(device).float()
            labels = labels.to(device)
        except StopIteration:  # Start new epoch if run out of data
            train_iter = iter(trainloader)
            batch, labels = next(train_iter)
            batch = batch.to(device).float()
            labels = labels.to(device)
        loss= train_step_fn(state, batch, labels)

        #writer.add_scalar("training_loss", loss.item(), step)

        # Save a temporary checkpoint to resume training if training is stopped
        if step != 0 and step % snapshot_freq_for_preemption == 0:
            print("Saving temporary checkpoint")
            save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        if step % eval_freq == 0:
            print("Starting evaluation")
            # Use 25 batches for test-set evaluation, arbitrary choice
            N_evals = 25
            for i in range(N_evals):
                try:
                    eval_batch, labels = next(eval_iter)
                    eval_batch = batch.to(device).float()
                    labels = labels.to(device)
                except StopIteration:  # Start new epoch
                    eval_iter = iter(testloader)
                    eval_batch, labels = next(eval_iter)
                    eval_batch = batch.to(device).float()
                    labels = labels.to(device)
                eval_loss = eval_step_fn(state, eval_batch,labels)
                eval_loss = eval_loss.detach()
            print("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))

        # Save a checkpoint periodically
        if step != 0 and step % snapshot_freq == 0 or step == num_train_steps:
            print("Saving a checkpoint")
            # Save the checkpoint.
            save_step = step // snapshot_freq
            save_checkpoint(os.path.join(
                checkpoint_dir, 'checkpoint_{}.pth'.format(save_step)), state)
        
        if step != 0 and step%sample_freq == 0 or step == num_train_steps:
            sample(step = str(step))




    
sample(step = str(int(state['step'])))
