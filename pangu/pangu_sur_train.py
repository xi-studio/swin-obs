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

    def preprocess(self, x):
        x[0] = (x[0] - 220.0) / (315.0 - 220.0)
        x[1] = (x[1]/100.0 - 950.0) / (1050.0 - 950.0)
        x[2] = (x[2] - (-30.0)) / (30.0 - (-30.0))
        x[3] = (x[3] - (-30.0)) / (30.0 - (-30.0))

        return x


    def __getitem__(self, index):

        sate = np.load(self.list[index][0]).astype(np.float32)
        pred = np.load(self.list[index][1]).astype(np.float32)
        obs  = np.load(self.list[index][2]).astype(np.float32)

        pred = pred[(3, 0, 1, 2), :, :]

        sate = np.nan_to_num(sate, nan=255)
        sate = (sate - 180.0) / (375.0 - 180.0)

        pred = self.preprocess(pred)
        obs  = self.preprocess(obs)

        sate = resize(sate, (10, 256, 256))
        obs  = resize(obs, (4, 256, 256))
        
        pred_input  = np.concatenate((sate, pred), axis=0)

        return pred_input, obs

    def __len__(self):
        return len(self.list)

class UNetModel(nn.Module):

    def __init__(self, config):
        super(UNetModel, self).__init__()

        self.n_channels   = config['in_channels']
        self.mul = config['mul_channels']

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

        self.inc = double_conv(self.n_channels, self.mul)
        self.down1 = down(self.mul, self.mul * 2)
        self.down2 = down(self.mul * 2, self.mul * 4)
        self.down3 = down(self.mul * 4, self.mul * 8)
        self.down4 = down(self.mul * 8, self.mul * 8)
        self.up0 = up(self.mul * 8, self.mul * 8) 
        self.up1 = up(self.mul * 16, self.mul * 4)
        self.up2 = up(self.mul * 8, self.mul * 2)
        self.up3 = up(self.mul * 4, self.mul)
        self.out = nn.Conv2d(self.mul * 2, self.n_channels - 10, kernel_size=1)

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


    best_acc = 1
    for epoch in range(epoch_num):
        model.train()
       
        with tqdm(total=len(train_loader)) as pbar:
            for i, (x, y) in enumerate(train_loader):
               out = model(x)
               loss = criterion(out, y)

               optimizer.zero_grad()
               accelerator.backward(loss)
               optimizer.step()

               pbar.set_description("train epoch[{}/{}] loss:{:.5f}".format(epoch + 1, epoch_num, loss))
               pbar.update(1)

        model.eval()
        accurate = 0
        num_elems = 0
        for _, (x, y) in enumerate(val_loader):
            with torch.no_grad():
                out = model(x)
                loss = criterion(out, y)
                num_elems += 1
                accurate += loss 
    
        eval_metric = accurate / num_elems
        accelerator.print(f"epoch {epoch}: {eval_metric:.5f}")
        if eval_metric < best_acc:
            best_acc = eval_metric
            output_dir = f"./logs/epoch_{epoch}"
            accelerator.save_state(output_dir)


def main(): 
    config = {"lr": 4e-5, "num_epochs": 500, "seed": 42, "batch_size": 16, "in_channels": 14, "mul_channels": 128}
    config['filenames'] = 'data/meta/pangu_01_train.npy'
    training_function(config)

if __name__ == '__main__':
    main()
