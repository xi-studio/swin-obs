import glob
import numpy as np
import time

res = glob.glob('../data/satelite/obs/*surface*.npy')

res.sort()

file_list = []
for x in res:
    upper = x.replace('surface', 'high')
    name = x.split("_")[-1]
    tname = name.split(".")[0]
    sate_time = time.strftime("%Y%m%d%H%M", time.strptime(tname, "%Y-%m-%dT%H"))

    base_file = '../data/satelite/china_himawari/H09_IR_%s_china_0p25.npy'
    
    sate_name = base_file % sate_time

    file_list.append([x[3:], upper[3:], sate_name[3:]])


np.save('../data/meta/obs_meta.npy', np.array(file_list))
print(file_list)
