import torch
import argparse
import os
import yaml
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

from sat_dataset import Radars
from unet import UNetModel


def training_function(args, config):
    epoch_num     = args.num_epoch
    batch_size    = args.batch_size 
    num_workers   = args.num_workers
    batch_size    = args.batch_size 
    fake          = args.fake
    log_time      = args.log_time
    learning_rate = config['lr']

    filenames     = np.arange(100) 

    if args.fake == False:
        filenames = np.load(args.filenames)

    dataset = Radars(filenames, fake) 
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = UNetModel(config)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    accelerator = Accelerator(log_with="all", project_dir='logs_sat')
    model, optimizer, test_loader = accelerator.prepare(model, optimizer, test_loader)

    accelerator.load_state('logs_sat/checkpoint_s2e_1225/best')
    for epoch in range(epoch_num):
        model.eval()
        accurate = 0
        num_elems = 0
       
        for i, (obs, sate) in enumerate(test_loader):
            with torch.no_grad():
                out = model(sate)
                loss = criterion(out, obs)
                
                np.save('pre_sur.npy', out.cpu().numpy())
                np.save('obs_sur.npy', obs.cpu().numpy())

                num_elems += 1
                accurate += loss 
                print(i)
                print('loss:', loss)
                break 
        eval_metric = accurate / num_elems
        accelerator.print(f"epoch {epoch}: {eval_metric:}")


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
