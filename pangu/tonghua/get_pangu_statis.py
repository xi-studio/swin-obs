import glob
import numpy as np
import time
import os

res = glob.glob('data/72/*/*.npz')

res.sort()

mlist = []
for i, x in enumerate(res):
    print(x)
    era = np.load(x)
    sur = era['obs_sur']
    upp = era['obs_upper']
    N, C, W, H = upp.shape
    upp = upp.reshape((N*C, W, H))
    obs = np.concatenate((sur, upp), axis=0)
    C, W, H = obs.shape
    obs = obs.reshape((C, W*H))
    mean = obs.mean(axis=1)
    std = obs.std(axis=1)
    mlist.append((mean, std))
    #if i>10:
    #  break

s = np.array(mlist)
mean = s[:, 0].mean(axis=0)
std = s[:, 1].mean(axis=0)

#print(mean.shape)
#print(std.shape)

np.savez('data/pangu_statis.npz', mean = mean, std = std)
