import numpy as np
import os
filename = 'data/meta/test_01.npy'

res = np.load(filename)
for i, x in enumerate(res):
    if os.path.isfile(x[0]) == False:
        print('x[0]:', x[0])
    if os.path.isfile(x[1]) == False:
        print('x[1]:', x[1])
    if os.path.isfile(x[2]) == False:
        print('x[2]:', x[2])
    print(i)
    np.load(x[0])
    np.load(x[1])
    np.load(x[2])
