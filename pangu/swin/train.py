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
            obs  = np.load(self.list[index][2][1:]).astype(np.float32)

            sate = np.nan_to_num(sate, nan=255)
            sate = (sate - 180.0) / (375.0 - 180.0)

            obs  = self.preprocess(obs)

            sate = resize(sate, (10, 256, 256))
            obs  = resize(obs, (4, 256, 256))
        else:
            sate = np.ones((10, 256, 256), dtype=np.float32)
            obs  = np.ones((4, 256, 256), dtype=np.float32)

        return sate, obs

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
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)

    model = UNetModel(config)
    #criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    accelerator = Accelerator(log_with="all", project_dir='logs_swin')
    hps = {"num_iterations": epoch_num, "learning_rate": learning_rate}
    accelerator.init_trackers("log_1218", config=hps)
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    best_acc = 1
    overall_step = 0
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

               overall_step += 1
               accelerator.log({"training_loss": loss}, step=overall_step)

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

        if epoch > 250:
            accelerator.save_state(f"./logs_swin/checkpoint_1218/epoch_{epoch}")
        if eval_metric < best_acc:
            best_acc = eval_metric
            accelerator.save_state("./logs_swin/checkpoint_1218/best")


def main(): 
    config = {"lr": 4e-5, 
              "num_epochs": 300, 
              "seed": 42, 
              "batch_size": 12, 
              "in_channels": 10, 
              "out_channels": 4, 
              "channels": 5, 
              "embed_dim": 96 * 2,
              "filenames": '../data/meta/train_pangu_24.npy',
              "fake": False,
              "depths": [2, 6, 18]
              }

    training_function(config)

if __name__ == '__main__':
    main()
