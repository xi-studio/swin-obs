import glob
import numpy as np
import time

res = glob.glob('data/train/sat_2020/*.npy')

res.sort()

file_list = []

for x in res[1:-20]:
    print(x)
    name = x.split("_")[-3]
    print(name[:-2])
    print('data/train/pred_2020/*_%s.npy' % name[:-2])
    pname = glob.glob('data/train/pred_2020/*_%s.npy' % name[:-2])
    oname = glob.glob('data/train/obs_2020/*_%s.npy' % name[:-2])
    print(pname)
    print(oname)


    if len(pname) != 0 and len(oname) != 0:
        file_list.append((x, pname[0], oname[0]))

np.save('data/meta/era5_meta.npy', np.array(file_list))
