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

class Radars(Dataset):
    def __init__(self, filenames):
        super(Radars, self).__init__()

        self.list = filenames 

    def __getitem__(self, index):

        satelite    = np.load(self.list[index][0]).astype(np.float32)
        pg  = np.load(self.list[index][1])
        pg_surface = (pg['surf'][0,0]).astype(np.float32)
        era_surface = np.load(self.list[index][2]).astype(np.float32)

        satelite = np.nan_to_num(satelite, nan=255)
        satelite = (satelite - 180.0) / (375.0 - 180.0)

        pg_tem = (pg_surface[0] - 220) / (315 - 220)
        era_tem = (era_surface[0] - 220) / (315 - 220)

        satelite = resize(satelite, (10, 256, 256))
        era_tem = resize(era_tem, (256, 256))
        
        pg_tem = pg_tem.reshape((1, 256, 256))
        era_tem = era_tem.reshape((1, 256, 256))
       
        pg_input  = np.concatenate((satelite, pg_tem),axis=0)

        return pg_input, era_tem 

    def __len__(self):
        return len(self.list)

class UNetModel(nn.Module):

    def __init__(self, config):
        super(UNetModel, self).__init__()

        self.n_channels   = config['in_channels']
        self.mul_channels = config['mul_channels']

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(32, out_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(32, out_channels),
                nn.SiLU(inplace=True),
            )

        def down(in_channels, out_channels):
            return nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_channels, out_channels)
            )
   
        def up(in_channels, out_channels):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                double_conv(in_channels, out_channels)
            )

        self.inc = double_conv(self.n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up0 = up(512, 512) 
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.out = nn.Conv2d(128, self.n_channels - 10, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up0(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.up1(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.out(x)

        return x 


def training_function(config):
    epoch_num     = config['num_epochs']
    batch_size    = config['batch_size']
    learning_rate = config['lr']
    filenames     = np.load(config['filenames'])
    
    dataset = Radars(filenames) 
    #n_val = int(len(dataset) * 0.1)
    #n_train = len(dataset) - n_val
    #train_ds, val_ds = random_split(dataset, [n_train, n_val])
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    #val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)


    model = UNetModel(config)
    #criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    accelerator = Accelerator()
    model, optimizer, test_loader = accelerator.prepare(model, optimizer, test_loader)


    for epoch in range(epoch_num):
        accelerator.load_state('logs/best_epoch')
        model.eval()
        accurate = 0
        base = 0
        num_elems = 0
        for _, (x, y) in enumerate(test_loader):
            with torch.no_grad():
                out = model(x)
                loss = criterion(out, y)
                out1 = x[:,10].unsqueeze(1)
                loss1 = criterion(out1, y)
                num_elems += 1
                accurate += loss 
                base += loss1
                print('loss', loss)
                print('loss1',loss1)
    
        eval_metric = accurate / num_elems
        base_metric = base / num_elems
        accelerator.print(f"epoch {epoch}: {eval_metric:.5f}")
        accelerator.print(f"epoch {epoch}: {base_metric:.5f}")


def main(): 
    config = {"lr": 4e-5, "num_epochs": 1, "seed": 42, "batch_size": 1, "in_channels": 11, "mul_channels":64}
    #config['filenames'] = 'data/meta/eval_01_meta.npy'
    config['filenames'] = 'data/meta/pred_01_meta.npy'
    training_function(config)

if __name__ == '__main__':
    main()
