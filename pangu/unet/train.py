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

from mydataset import Radars
from unet import UNetModel


def training_function(config):
    epoch_num     = config['num_epochs']
    batch_size    = config['batch_size']
    learning_rate = config['lr']
    filenames     = np.load(config['filenames'])
    fake          = config['fake']
    log_time      = config['log_time']
    
    dataset = Radars(filenames, fake) 
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)

    model = UNetModel(config)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    accelerator = Accelerator(log_with="all", project_dir='logs_sat')
    hps = {"num_iterations": epoch_num, "learning_rate": learning_rate}
    accelerator.init_trackers(f"log_{log_time}" , config=hps)
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
            accelerator.save_state(f"./logs_sat/checkpoint_{log_time}/epoch_{epoch}")
        if eval_metric < best_acc:
            best_acc = eval_metric
            accelerator.save_state(f"./logs_sat/checkpoint_{log_time}/best")


def main(): 
    config = {"lr": 4e-5, 
              "num_epochs": 300, 
              "seed": 42, 
              "batch_size": 4, 
              "in_channels": 69, 
              "out_channels": 10, 
              "mul_channels": 96, 
              "filenames": '../data/meta/train_pangu_24.npy',
              "fake": True,
              "log_time": '1221'
              }

    training_function(config)

if __name__ == '__main__':
    main()
