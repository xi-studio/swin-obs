import numpy as np

res = np.load('./data/meta/era5_to_sat_train.npy')

mean_list = []
std_list  = [] 

for i, f in enumerate(res):
    obs = np.load('./data/train/era5_obs_2020/'+f[1])
    x = obs['surface']
    y = obs['upper']

    N, C, W, H = y.shape
    y = y.reshape((N*C, W, H))
    obs = np.concatenate((x, y), axis=0)
    C, W, H = obs.shape


    obs = obs.reshape((C, W*H))
    mean = np.mean(obs, axis=1)
    std  = np.std(obs, axis=1)
    

    mean_list.append(mean)
    std_list.append(std)

    print(i)

a = np.array(mean_list)
b = np.array(std_list)

np.savez('./data/era5_statis.npz', mean=np.mean(a, axis=0), std=np.mean(b, axis=0))

