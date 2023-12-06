import glob
import numpy as np
import time
from skimage.transform import rescale, resize
res = glob.glob('../data/satelite/02_pred/*.npz')

res.sort()

file_list = []

num = 0
sum_L1_loss = 0
for x in res[:100]:
    print(x)
    name = x.split("_")[-1]
    tname = name.split(".")[0]
    #print(tname)
    era5_time = time.strftime("%Y-%m-%dT%H", time.strptime(tname, "%Y%m%d%H"))
    #print(era5_time)

    era5_surface = 'data/satelite/obs_02/surface_%s.npy' % era5_time

    pred = np.load(x)
    pred_surface = pred['surf'][0,0]
    era_surface = np.load('../'+ era5_surface)


    print(pred_surface.shape)
    print(era_surface.shape)

    pg_tem = (pred_surface[0] - 220.0) /(315.0 - 220.0)
    pg_tem = resize(pg_tem, (241,281))
    era_tem = (era_surface[0] - 220.0) /(315.0 - 220.0)
     
    #print(pg_tem)
    #print(era_tem)
    mse = np.mean((pg_tem - era_tem)**2)
    L1 = np.mean(np.abs(pg_tem - era_tem))
    
    sum_L1_loss += L1
    num += 1
    print(mse, L1)

print("ALL:", sum_L1_loss/num)


