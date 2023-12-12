import glob
import numpy as np
import time

res = glob.glob('../data/train/sat_2020/*.npy')

res.sort()

file_list = []

for x in res[24:-24]:
    print(x)
    name = x.split("_")[-3]
    print(name)
    now   = time.mktime(time.strptime(name, "%Y%m%d%H%M"))
    tname = time.strftime("%Y-%m-%dT%H", time.localtime(now - 3600 * 24))
    pname = glob.glob('../data/train/24_output/*_%s.npy' % tname)
    oname = glob.glob('../data/train/obs_2020/*_%s.npy' % name[:-2])
    print(pname)
    print(oname)


    if len(pname) != 0 and len(oname) != 0:
        file_list.append((x, pname[0], oname[0]))

np.save('../data/meta/pangu_meta.npy', np.array(file_list))
