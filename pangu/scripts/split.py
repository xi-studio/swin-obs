import numpy as np
import time

#res = np.load('../data/meta/era5_meta.npy')
res = np.load('../data/meta/pangu_meta.npy')
print(res.shape)

train_list = []
val_list = []
for x in res:
    name = time.strptime(x[0].split('_')[-3], "%Y%m%d%H%M")
    if name.tm_mday < 28:
        train_list.append(x)
    else:
        val_list.append(x)

print(len(train_list))
print(len(val_list))

np.save('../data/meta/train_pangu_24.npy', np.array(train_list))
np.save('../data/meta/test_pangu_24.npy', np.array(val_list))
