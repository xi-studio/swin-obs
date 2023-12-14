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

        pred_input  = np.concatenate((sate, pred), axis=0)

        return pred_input, obs

    def __len__(self):
        return len(self.list)


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

    model = SwinTransformer3D(in_chans=14, 
                              patch_size=(2,4,4), 
                              embed_dim=48, 
                              window_size=(2,7,7), 
                              depths=[2, 2, 2]
                              )
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    accelerator = Accelerator(log_with="all", project_dir='logs_swin')
    hps = {"num_iterations": epoch_num, "learning_rate": learning_rate}
    accelerator.init_trackers("log_1214", config=hps)
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
            accelerator.save_state(f"./logs_swin/checkpoint_1214/epoch_{epoch}")
        if eval_metric < best_acc:
            best_acc = eval_metric
            accelerator.save_state("./logs_swin/checkpoint_1214/best")


def main(): 
    config = {"lr": 4e-5, "num_epochs": 300, "seed": 42, "batch_size": 16, "in_channels": 14}
    config['filenames'] = '../data/meta/train_pangu_24.npy'
    training_function(config)

if __name__ == '__main__':
    main()
