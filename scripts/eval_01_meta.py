import glob
import numpy as np
import time
from skimage.transform import rescale, resize
res = glob.glob('../data/satelite/02_pred/*.npz')


res.sort()

file_list = []

for x in res[:24*10]:
    print(x)
    name = x.split("_")[-1]
    tname = name.split(".")[0]
    #print(tname)
    era5_time = time.strftime("%Y-%m-%dT%H", time.strptime(tname, "%Y%m%d%H"))
    sat_time = time.strftime("%Y%m%d%H%M", time.strptime(tname, "%Y%m%d%H"))
    #print(era5_time)

    satelite = 'data/satelite/china_himawari/H09_IR_%s_china_0p25.npy' % sat_time 
    era5_surface = 'data/satelite/obs_02/surface_%s.npy' % era5_time

    file_list.append((satelite, x[3:], era5_surface))

np.save('../data/meta/eval_01_meta.npy', np.array(file_list))
print(file_list)
