import glob
import numpy as np
import time

res = glob.glob('../data/obs_2020/*.npy')
res.sort()
file_list = []

for x in res[24:-24]:
    print(x)
    name = x.split("_")[-1]
    tname = x.split(".")[0]
    print(tname)
   
    now   = time.mktime(time.strptime(tname, "%Y-%m-%dT%H"))
    tname_1  = time.strftime("%Y-%m-%dT%H", time.localtime(now - 3600))
    tname_6  = time.strftime("%Y-%m-%dT%H", time.localtime(now - 3600 * 6))
    tname_24 = time.strftime("%Y-%m-%dT%H", time.localtime(now - 3600 * 24))
    
    filename_1  = '../data/1_output/surface_1_%s.npy' % tname_1
    filename_6  = '../data/6_output/surface_6_%s.npy' % tname_6
    filename_24 = '../data/24_output/surface_24_%s.npy' % tname_24



    file_list.append((filename_1, filename_6, filename_24))

np.save('data/meta/pangu_meta.npy', np.array(file_list))
