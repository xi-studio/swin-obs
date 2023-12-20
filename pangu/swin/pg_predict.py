import torch
import argparse
import os
import re
from tqdm import tqdm
from skimage.transform import rescale, resize

import numpy as np
import PIL
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor, Compose
import torchvision.datasets as datasets
from accelerate import Accelerator
from torchvision.transforms import Compose, RandomResizedCrop, Resize, ToTensor

from swin_3d_model import SwinTransformer3D 


class Radars(Dataset):
    def __init__(self, filenames, fake=False):
        super(Radars, self).__init__()

        self.list = filenames 
        self.fake = fake

    def preprocess(self, x):
        x[0] = (x[0] - 220.0) / (315.0 - 220.0)
        x[1] = (x[1]/100.0 - 950.0) / (1050.0 - 950.0)
        x[2] = (x[2] - (-30.0)) / (30.0 - (-30.0))
        x[3] = (x[3] - (-30.0)) / (30.0 - (-30.0))

        return x


    def __getitem__(self, index):


        if self.fake!=True:
            sate = np.load(self.list[index][0][1:]).astype(np.float32)
            pred = np.load(self.list[index][1][1:]).astype(np.float32)
            obs  = np.load(self.list[index][2][1:]).astype(np.float32)

            pred = pred[(3, 0, 1, 2), :, :]

            sate = np.nan_to_num(sate, nan=255)
            sate = (sate - 180.0) / (375.0 - 180.0)

            pred = self.preprocess(pred)
            obs  = self.preprocess(obs)

            sate = resize(sate, (10, 256, 256))
            pred = resize(pred, (4, 256, 256))
            obs  = resize(obs, (4, 256, 256))

            pred_input = np.concatenate((sate, pred), axis=0)
        else:
            pred_input = np.ones((14, 256, 256), dtype=np.float32)
            obs  = np.ones((4, 256, 256), dtype=np.float32)

        return pred_input, obs

    def __len__(self):
        return len(self.list)
class UNetModel(nn.Module):

    def __init__(self, config):
        super(UNetModel, self).__init__()

        self.in_chans = config['in_channels']
        self.out_chans = config['out_channels']
        self.chans = config['channels']
        self.dim = config['embed_dim']
        self.depths = config['depths']

        
        self.swin3d = SwinTransformer3D(in_chans=self.chans, 
                              patch_size=(2,4,4), 
                              embed_dim=self.dim, 
                              window_size=(2,7,7), 
                              depths=self.depths
                              )
        self.out = nn.Sequential(
                nn.Conv2d(self.in_chans, self.dim, kernel_size=1),
                nn.Conv2d(self.dim, self.out_chans, kernel_size=1)
        )

    def forward(self, x):
        x = self.swin3d(x)
        x = self.out(x)

        return x 

def training_function(config):
    epoch_num     = config['num_epochs']
    batch_size    = config['batch_size']
    learning_rate = config['lr']
    filenames     = np.load(config['filenames'])
    fake          = config['fake']
    
    dataset = Radars(filenames, fake) 
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    model = UNetModel(config)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    accelerator = Accelerator(log_with="all", project_dir='logs_pangu')
    model, optimizer, test_loader = accelerator.prepare(model, optimizer, test_loader)

    accelerator.load_state('logs_pangu/checkpoint_1219/best')
    for epoch in range(epoch_num):
        model.eval()
        accurate = 0
        num_elems = 0
        for i, (x, y) in enumerate(test_loader):
            with torch.no_grad():
                out = model(x)
                loss = criterion(out[0,2], y[0,2])

                num_elems += 1
                accurate += loss 
                print(i)
                print('loss:', loss)
    
        eval_metric = accurate / num_elems
        accelerator.print(f"epoch {epoch}: {eval_metric:}")
       


def main(): 
    config = {"lr": 4e-5, 
              "num_epochs": 1, 
              "seed": 42, 
              "batch_size": 1, 
              "in_channels": 14, 
              "out_channels": 4, 
              "channels": 7, 
              "embed_dim": 96 * 2,
              "filenames": '../data/meta/test_pangu_24.npy',
              "fake": False,
              "depths": [2, 6, 18]
              }

    training_function(config)

if __name__ == '__main__':
    main()
