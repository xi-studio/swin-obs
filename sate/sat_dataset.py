import torch
import os
from skimage.transform import rescale, resize
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor, Compose
import torchvision.datasets as datasets
from torchvision.transforms import Compose, RandomResizedCrop, Resize, ToTensor
import gcsfs

class Radars(Dataset):
    def __init__(self, filenames, fake=False):
        super(Radars, self).__init__()

        self.list = filenames 
        self.fake = fake

    def load_sat(self, filename):
        sate = np.load(filename)
        sate = np.nan_to_num(sate, nan=255)
        sate = (sate - 180.0) / (375.0 - 180.0)

        return sate


    def __getitem__(self, index):

        if self.fake!=True:
            t0_name = './data/train/' + self.list[index][0]
            t1_name = './data/train/' + self.list[index][1]
            t0 = self.load_sat(t0_name).astype(np.float32)
            t1 = self.load_sat(t1_name).astype(np.float32)

            t0 = resize(t0, (10, 256, 256))
            t1 = resize(t1, (10, 256, 256))
        else:
            t0 = np.ones((10, 256, 256), dtype=np.float32)
            t1 = np.ones((10, 256, 256), dtype=np.float32)

        return t0, t1

    def __len__(self):
        return len(self.list)

if __name__ == '__main__':
    filename = np.load('data/meta/sat_pred.npy')
    a = Radars(filenames=filename, fake=False)

    train_loader = DataLoader(a, batch_size=1, shuffle=True, num_workers=4)
    for x in train_loader:
        print(x[0].shape, x[1].shape)
        break
