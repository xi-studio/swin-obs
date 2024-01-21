import glob
import numpy as np
import time
import os

res = glob.glob('data/china_fp16/*.npz')

res.sort()

base_dir = 'data/china_himawari/H09_IR_%s_china_0p25.npy'

mlist = []
for x in res:
    print(x)
    name = x.split('/')[-1]
    #print(name[:-4])

    r = time.strptime(name[:-4], '%Y-%m-%dT%H')
    sname = time.strftime('%Y%m%d%H%M', r)
    print(sname)
    filename = base_dir % sname
    if os.path.isfile(filename):
        #y = np.load(filename)
        #print(y.shape)
        mlist.append((x, filename))

np.save('data/pan_fp16_meta.npy', np.array(mlist))
