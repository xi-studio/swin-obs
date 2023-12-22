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

    def preprocess(self, x, y):
        x[0] = (x[0] - 220.0) / (315.0 - 220.0)         #temp
        x[1] = (x[1]/100.0 - 950.0) / (1050.0 - 950.0)  #mslp
        x[2] = (x[2] - (-30.0)) / (30.0 - (-30.0))      #wind_u
        x[3] = (x[3] - (-30.0)) / (30.0 - (-30.0))      #wind_v

        y[0] = (y[0] - 0.0) / (160.0 - 0.0)             #geopotential
        y[1] = (y[1] - 185.0) / (315.0 - 185.0)         #temperature
        y[2] = (y[2] - 0.0) / (0.02 - 0.0)              #specific_humidity
        y[3] = (y[3] - (-60)) / (60.0 - (-60.0))         #u_component_of_wind
        y[4] = (y[4] - (-60)) / (60.0 - (-60.0))         #v_component_of_wind
        
        obs = np.concatenate((x, y), axis=0)

        return obs

    def load_sat(self, filename):
        fs = gcsfs.GCSFileSystem(project='era5_jy_test')
        with fs.open(filename,'rb') as f:
            sate = np.load(f)
            sate = np.nan_to_num(sate, nan=255)
            sate = (sate - 180.0) / (375.0 - 180.0)

            return sate

    def load_obs(self, filename):
        fs = gcsfs.GCSFileSystem(project='era5_jy_test')
        with fs.open(filename,'rb') as f:
            res = np.load(f)
            sur = res['surface']
            upp = res['upper']

            N, C, W, H = upp.shape
            upp = upp.reshape((N*C, W, H))
            obs  = self.preprocess(sur, upp)

            return obs


    def __getitem__(self, index):

        if self.fake!=True:
            satename = 'himawari-caiyun/china_himawari/' + self.list[index][0]
            obsname  = 'era5_jy_test/pangu/era5_obs_2020/' + self.list[index][1]
            sate = self.load_sat(satename)
            obs  = self.load_obs(obsname)

            sate = resize(sate, (10, 256, 256))
            obs  = resize(obs, (69, 256, 256))
        else:
            sate = np.ones((10, 256, 256), dtype=np.float32)
            obs  = np.ones((69, 256, 256), dtype=np.float32)

        return obs, sate

    def __len__(self):
        return len(self.list)

if __name__ == '__main__':

    #filename = np.load('data/meta/era5_to_sat_train.npy')
    filename = np.arange(100)
    a = Radars(filenames=filename, fake=True)

    train_loader = DataLoader(a, batch_size=1, shuffle=True, num_workers=4)
    for x in train_loader:
        print(x[0].shape, x[1].shape)
        break
