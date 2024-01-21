import glob
import numpy as np
import time
import os

res = glob.glob('data/china_fp16/*.npz')

res.sort()

mlist = []
for i, x in enumerate(res):
    print(x)
    era = np.load(x)
    sur = era['sur']
    upp = era['upper']
    sur = sur.astype(np.float32)
    upp = upp.astype(np.float32)
    N, C, W, H = upp.shape
    upp = upp.reshape((N*C, W, H))
    obs = np.concatenate((sur, upp), axis=0)
    C, W, H = obs.shape
    obs = obs.reshape((C, W*H))
    mean = obs.mean(axis=1)
    #dmax = obs.max(axis=1)
    #dmin = obs.min(axis=1)
    std = obs.std(axis=1)
    mlist.append((mean, std))
    #if i>10:
    #  break

s = np.array(mlist)
mean = s[:, 0].mean(axis=0)
std = s[:, 1].mean(axis=0)

print(mean)
print(std)

np.savez('data/pangu_fp16_statis.npz', mean = mean, std = std)
