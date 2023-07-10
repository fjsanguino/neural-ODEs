# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 21:36:31 2023

@author: nick8
"""

import os
import torch


from model import get_model
from numpy import random
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from sklearn.metrics import accuracy_score

from torch.utils.tensorboard import SummaryWriter

MODEL_NAME = 'Paper2'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 28
num_classes = 10
img_channels = 1
BATCH_SIZE = 32
SAVE_DIR = os.path.join('runs','denoising' ,MODEL_NAME, '_lr0001')
EPOCH = 200
LR = 0.001
SAMPLE_RATE = 10

def correct_name(name = MODEL_NAME):
   if name == 'Paper':
      return 'Paper2'
   elif name == 'MLP':
      return 'MLP2'
   else:
      return name
MODEL_NAME = correct_name(MODEL_NAME)

def seed_init_fn(x):
   #seed = args.seed + x
   seed = x
   np.random.seed(seed)
   random.seed(seed)
   torch.manual_seed(seed)
   return

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def sample(model, testloader, epoch = None, save_dir = SAVE_DIR):
   if epoch is None:
      epoch = 0
   sample_img, label = next(iter(testloader))
   
   noisy_img = add_noise(sample_img)
   
   pred = model(noisy_img).detach().numpy()
   sample_dir = os.path.join(save_dir,"samples")
   
   if not os.path.exists(sample_dir):
       os.makedirs(sample_dir)
   
   for i in range(0,6):
        
       plt.subplot(3,6,i+1)
       plt.tight_layout()
       plt.imshow(sample_img[i][0], cmap='gray', interpolation='none')
       plt.title("original")
       plt.xticks([])
       plt.yticks([])
       
   for i in range(0,6):
       
       plt.subplot(3,6,i+7)
       plt.tight_layout()
       plt.imshow(pred[i][0], cmap='gray', interpolation='none')
       plt.title("denoised")
       plt.xticks([])
       plt.yticks([])
       
   for i in range(0,6):
       
       plt.subplot(3,6,i+13)
       plt.tight_layout()
       plt.imshow(noisy_img[i][0], cmap='gray', interpolation='none')
       plt.title("noisy")
       plt.xticks([])
       plt.yticks([])
       
       
       
   plt.savefig(os.path.join(sample_dir,"sample_epoch_"+str(epoch)))
   


def evaluate(model, data_loader):
    ''' set model to evaluate mode '''
    model.eval()
    accs = []

    with torch.no_grad():  # do not need to caculate information for gradient during eval
        for idx, (imgs, gt) in enumerate(data_loader):
            imgs = imgs.to(DEVICE)
            
            noisy_img = add_noise(imgs)
            
            
            
            pred = model(noisy_img)
            loss = nn.MSELoss()
            acc = loss(imgs,pred)
            
            accs.append(acc)

    
    accs = np.array(accs)

    return np.mean(accs)

def add_noise(image):
    noise = torch.randn_like(image)*0.3

    return torch.clip(image + noise,0.,1.)

if __name__ == '__main__':


    '''create directory to save trained model and other info'''
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    ''' setup DEVICE'''

    ''' setup random seed '''
    seed_init_fn(0)

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    """Instantiate dataloaders of training and test datasets"""
    transform = [transforms.Resize(IMG_SIZE),
                 transforms.CenterCrop(IMG_SIZE), transforms.ToTensor()]
    transform = transforms.Compose(transform)
    training_data = datasets.MNIST(
        root=".", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(
        root=".", train=False, download=True, transform=transform)

    train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    ''' load model '''
    print('===> prepare model ...')
    model = get_model(MODEL_NAME, input_dim = IMG_SIZE, output_dim = num_classes, in_channels = img_channels)
    
    model.to(DEVICE)  # load model to gpu

    ''' define loss '''
    criterion = nn.MSELoss()

    ''' setup optimizer '''
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(SAVE_DIR, 'train_info'))

    ''' train model '''
    print('===> start training ...')
    iters = 0
    best_acc = 0
    
    
    for epoch in range(1, EPOCH + 1):

        model.train()

        for idx, (imgs, cls) in enumerate(train_loader):
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(train_loader))
            iters += 1

            ''' move data to gpu '''
            imgs, cls = imgs.to(DEVICE), cls.to(DEVICE)
            
            clean_img = imgs.clone()
            
            noisy_img = imgs.clone()
            ''' forward path '''
            output = model(noisy_img)

            ''' compute loss, backpropagation, update parameters '''
            loss = criterion(output, clean_img)  # compute loss

            optimizer.zero_grad()  # set grad of all parameters to zero
            loss.backward()  # compute gradient for each parameters
            optimizer.step()  # update parameters

            ''' write out information to tensorboard '''
            writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
            train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

            print(train_info)
        if epoch % SAMPLE_RATE == 0:
            sample(epoch = epoch, model = model, testloader = val_loader)

           
        if epoch % 1 == 0:
            ''' evaluate the model '''
            acc = evaluate(model, val_loader)
            writer.add_scalar('val_acc', acc, epoch)
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))

            ''' save best model '''
            if acc > best_acc:
                save_model(model, os.path.join(SAVE_DIR, 'model_best.pth.tar'))
                best_acc = acc

        ''' save model '''
        #save_model(model, os.path.join(SAVE_DIR, 'model_{}.pth.tar'.format(epoch)))

    print('Best acc is {}'.format(best_acc))
