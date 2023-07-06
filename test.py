import torch
import os
import numpy as np
from numpy import random
from sklearn.metrics import accuracy_score
from torchvision import datasets
from torchvision import transforms
from model import get_model
from torch.utils.data import DataLoader


MODEL_NAME = 'Paper'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 28
BATCH_SIZE = 32
SAVE_DIR = os.path.join('runs', MODEL_NAME)
EPOCH = 200
LR = 0.005
SAMPLE_RATE = 5


def seed_init_fn(x):
    # seed = args.seed + x
    seed = x
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

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

    ''' setup DEVICE'''

    ''' setup random seed '''
    seed_init_fn(0)

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    """Instantiate dataloaders of training and test datasets"""
    transform = [transforms.Resize(IMG_SIZE),
                 transforms.CenterCrop(IMG_SIZE), transforms.ToTensor()]
    transform = transforms.Compose(transform)

    test_data = datasets.MNIST(
        root=".", train=False, download=True, transform=transform)

    val_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    ''' load model '''
    print('===> prepare model ...')
    model = get_model(MODEL_NAME).to(device=DEVICE)

    ''' resume save model '''
    checkpoint = torch.load(os.path.join(SAVE_DIR, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint)

    acc = evaluate(model, val_loader)
    print('Best acc is {}'.format(acc))
