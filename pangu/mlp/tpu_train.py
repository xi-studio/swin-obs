import torch
import argparse
import os
import re
import yaml
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

from tpu_dataset import Radars
from model import UNetModel


def training_function(args, config):
    epoch_num     = args.num_epoch
    batch_size    = args.batch_size 
    num_workers   = args.num_workers
    batch_size    = args.batch_size 
    fake          = args.fake
    log_time      = args.log_time
    learning_rate = config['lr']

    filenames     = np.arange(3000) 

    if args.fake == False:
        filenames = np.load(args.filenames)
    
    train_ds = Radars(filenames[ : -2000], fake) 
    val_ds = Radars(filenames[-2000 : -1000], fake)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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

        if eval_metric < best_acc:
            best_acc = eval_metric
            accelerator.save_model(model, f"./logs_sat/checkpoint_{log_time}/best", safe_serialization=False)


def main(): 
    parser = argparse.ArgumentParser('Era5 to satelite Unet', add_help=True)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument('--batch_size', type=int, default=1, help="batch size for single GPU")
    parser.add_argument('--num_epoch', type=int, default=2, help="epochs")
    parser.add_argument('--num_workers', type=int, default=8, help="num workers")
    parser.add_argument('--log_time', type=str, default='1222', help="log time name")
    parser.add_argument('--filenames', type=str, help="data filenames")
    parser.add_argument('--fake', type=bool, default=False, help="if fake data")
    args, unparsed = parser.parse_known_args()

    with open(args.cfg, "r") as f:
        res = yaml.load(f, Loader=yaml.FullLoader)

    config = res[0]['config']

    training_function(args, config)

if __name__ == '__main__':
    main()
