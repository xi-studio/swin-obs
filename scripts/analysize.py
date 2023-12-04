import glob
import numpy as np

res = glob.glob("../data/satelite/china_himawari/*.npy")
res.sort()

max_list = []
min_list = []
mean_list = []
for i in res:
    x = np.load(i)
    x1 = np.nan_to_num(x)
    k = np.mean(x1)
    x2 = np.nan_to_num(x, nan = k)

    max_list.append(np.max(x2))
    min_list.append(np.min(x2))
    mean_list.append(np.mean(x2))

print(np.max(np.array(max_list)))
print(np.min(np.array(min_list)))
print(np.mean(np.array(mean_list)))
    
