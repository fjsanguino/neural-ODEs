import os
import torch

import model
from numpy import random

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from sklearn.metrics import accuracy_score

from torch.utils.tensorboard import SummaryWriter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 28
BATCH_SIZE = 32
SAVE_DIR = os.path.join('runs','MLP')
EPOCH = 200
LR = 0.001
sample_rate = 5

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
    pred = model(sample_img)
    for i in range(0,6):
        
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(sample_img[i][0], cmap='gray', interpolation='none')
        plt.title(torch.argmax(pred[i]))
        plt.xticks([])
        plt.yticks([])
    plt.savefig(os.path.join(save_dir,"samples","sample_epoch_"+str(epoch)))
   


def evaluate(model, data_loader):
    ''' set model to evaluate mode '''
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():  # do not need to caculate information for gradient during eval
        for idx, (imgs, gt) in enumerate(data_loader):
            imgs = imgs.to(DEVICE)
            pred = model(imgs)

            _, pred = torch.max(pred, dim=1)

            pred = pred.cpu().numpy().squeeze()
            gt = gt.numpy().squeeze()

            preds.append(pred)
            gts.append(gt)

    gts = np.concatenate(gts)
    preds = np.concatenate(preds)

    return accuracy_score(gts, preds)

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
    model = model.PaperModel()
    model.to(DEVICE)  # load model to gpu

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()

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

            ''' forward path '''
            output = model(imgs)

            ''' compute loss, backpropagation, update parameters '''
            loss = criterion(output, cls)  # compute loss

            optimizer.zero_grad()  # set grad of all parameters to zero
            loss.backward()  # compute gradient for each parameters
            optimizer.step()  # update parameters

            ''' write out information to tensorboard '''
            writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
            train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

            print(train_info)
        if epoch % sample_rate == 0:
            sample(epoch = epoch, model = model, testloader = val_loader)

           
        if epoch % 1 == 0:
            ''' evaluate the model '''
            acc = evaluate(model, val_loader)
            writer.add_scalar('val_acc', acc, iters)
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))

            ''' save best model '''
            if acc > best_acc:
                save_model(model, os.path.join(SAVE_DIR, 'model_best.pth.tar'))
                best_acc = acc

        ''' save model '''
        #save_model(model, os.path.join(SAVE_DIR, 'model_{}.pth.tar'.format(epoch)))

    print('Best acc is {}'.format(best_acc))
