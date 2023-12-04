import glob
import numpy as np
import time

res = glob.glob('../data/satelite/24_output/*surface*.npy')

res.sort()

file_list = []

for x in res:
    print(x)
    name = x.split("_")[-1]
    tname = name.split(".")[0]
    now = time.mktime(time.strptime(tname, "%Y-%m-%dT%H")) + 24*60*60
    era5_time = time.strftime("%Y-%m-%dT%H", time.localtime(now))
    sate_time = time.strftime("%Y%m%d%H%M", time.localtime(now))
    
    satelite = 'data/satelite/china_himawari/H09_IR_%s_china_0p25.npy' % sate_time
    pangu_surface = x
    pangu_upper = x.replace('surface', 'upper')

    era5_surface = 'data/satelite/obs/surface_%s.npy' % era5_time
    era5_high = 'data/satelite/obs/high_%s.npy' % era5_time

    print(pangu_surface)
    pg_surface = np.load(pangu_surface)
    era_surface = np.load('../'+ era5_surface)

    pg_tem = (pg_surface[0] - 220.0) /(315.0 - 220.0)
    print(pg_surface[0])
    print(np.max(pg_tem))
    print(np.min(pg_tem))
    era_tem = (era_surface[0] - 220.0) /(315.0 - 220.0)
     
    #print(pg_tem)
    #print(era_tem)
    #mse = np.mean((pg_tem - era_tem)**2)
    #print(mse)
    break
    #file_list.append((satelite, pangu_surface[3:], pangu_upper[3:], era5_surface, era5_high))

#np.save('../data/meta/era5_meta.npy', np.array(file_list))
#print(file_list)
