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
        statis = np.load('data/pangu_fp16_statis.npz')
        std = statis['std']
        std[std == 0.0] = 1
        m = np.ones((241*281, 69)) * statis['mean']
        s = np.ones((241*281, 69)) * std 
        self.mean = (m.T).reshape((69, 241, 281))
        self.std  = (s.T).reshape((69, 241, 281))


    def load_sat(self, filename):
        sate = np.load(filename)
        sate = np.nan_to_num(sate, nan=255)
        sate = (sate - 180.0) / (375.0 - 180.0)

        return sate

    def load_obs(self, filename):
        res = np.load(filename)
        sur = res['sur']
        upp = res['upper']
 
        N, C, W, H = upp.shape
        upp = upp.reshape((N*C, W, H))

        obs = np.concatenate((sur, upp), axis=0)
        obs = (obs - self.mean) / self.std 
 
        return obs
 

    def __getitem__(self, index):

        if self.fake!=True:
            obsname  = self.list[index][0]
            satename = self.list[index][1]
            sate = self.load_sat(satename)
            obs  = self.load_obs(obsname)

            sate = resize(sate, (10, 256, 256))
            obs  = resize(obs, (69, 256, 256))

            sate = sate.astype(np.float16)
            obs  = obs.astype(np.float16)
        else:
            sate = np.ones((10, 256, 256), dtype=np.float32)
            obs  = np.ones((69, 256, 256), dtype=np.float32)

        return obs, sate

    def __len__(self):
        return len(self.list)

if __name__ == '__main__':
    filename = np.load('data/pan_fp16_meta.npy')
    a = Radars(filenames=filename, fake=False)

    train_loader = DataLoader(a, batch_size=1, shuffle=True, num_workers=4)
    for x in train_loader:
        print(x[0].shape, x[1].shape)
        print(x[0].dtype, x[1].dtype)
        break
