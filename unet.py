import torch
import argparse
import os
import re
from tqdm import tqdm

import numpy as np
import PIL
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor, Compose
import torchvision.datasets as datasets
from accelerate import Accelerator
from torchvision.transforms import Compose, RandomResizedCrop, Resize, ToTensor

class Radars(Dataset):
    def __init__(self, filenames, transform=None):
        super(Radars, self).__init__()

        self.list = filenames 
        self.transform = transform

    def __getitem__(self, index):

        satelite    = np.load(self.list[index][0]).astype(np.float32)
        pg_surface  = np.load(self.list[index][1]).astype(np.float32)
        pg_upper    = np.load(self.list[index][2]).astype(np.float32)
        era_surface = np.load(self.list[index][3]).astype(np.float32)
        era_upper   = np.load(self.list[index][4]).astype(np.float32)
        

        pg_upper  = pg_upper.reshape((65, 241, 281))
        era_upper = era_upper.reshape((65, 241, 281))
       
        pg_input   = np.concatenate((satelite, pg_surface, pg_upper),axis=0)
        era_output = np.concatenate((era_surface, era_upper),axis=0)

        return pg_input, era_output 

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
        x  = nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
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
        x = nn.functional.interpolate(x, size=(241, 281), mode='bilinear', align_corners=True)

        return x 


def training_function(config):
    epoch_num     = config['num_epochs']
    batch_size    = config['batch_size']
    learning_rate = config['lr']
    filenames     = np.load(config['filenames'])
    
    train_tfm = Compose([ToTensor()])
    dataset = Radars(filenames, transform=train_tfm) 
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)


    model = UNetModel(config)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    accelerator = Accelerator()
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

       
    for epoch in range(epoch_num):
        model.train()
       
        with tqdm(total=len(train_loader)) as pbar:
            for i, (x_train, y_train) in enumerate(train_loader):
               out = model(x_train)
               loss = criterion(out, y_train)

               optimizer.zero_grad()
               accelerator.backward(loss)
               optimizer.step()

               pbar.set_description("train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epoch_num, loss))
               pbar.update(1)

def main(): 
    config = {"lr": 4e-5, "num_epochs": 3, "seed": 42, "batch_size": 1, "in_channels": 79, "mul_channels":128}
    config['filenames'] = 'data/meta/era5_meta.npy'
    training_function(config)

if __name__ == '__main__':
    main()
