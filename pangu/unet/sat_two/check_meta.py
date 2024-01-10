import time
import numpy as np
import os

res = np.load('data/meta/era5_to_sat_train.npy')

base = './data/train/sat_2020/'
my_list = []

for i, x in enumerate(res):
    name = x[0].split('_')[2]
    #print(name)
    now = time.mktime(time.strptime(name, "%Y%m%d%H%M"))
    tname = time.strftime("%Y%m%d%H%M", time.localtime(now - 3600))
    #print(tname)

    tname = (x[0]).replace(name, tname)
    t0 = base + tname
    #print(t0)
    if os.path.isfile(t0):
        print(t0)
        np.load(t0)
        my_list.append((tname, x[0], x[1]))
#np.save('./data/meta/sat_two_train.npy', np.array(my_list))
    

