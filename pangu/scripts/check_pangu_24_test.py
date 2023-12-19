import numpy as np
import os

def preprocess(x):
    x[0] = (x[0] - 220.0) / (315.0 - 220.0)
    x[1] = (x[1]/100.0 - 950.0) / (1050.0 - 950.0)
    x[2] = (x[2] - (-30.0)) / (30.0 - (-30.0))
    x[3] = (x[3] - (-30.0)) / (30.0 - (-30.0))

    return x

def check_MAE(x, pred):
    tem = np.mean(np.abs(pred[0] - x[0]))
    mslp   = np.mean(np.abs(pred[1] - x[1]))
    u = np.mean(np.abs(pred[2] - x[2]))
    v = np.mean(np.abs(pred[3] - x[3]))
    cat = np.mean(np.abs(pred - x) / x)

    return (tem, mslp, u, v, cat)


def main():
    filename = '../data/meta/test_pangu_24.npy'
    res = np.load(filename)

    list_24 = []

    for i, x in enumerate(res):
        print(i)
        pred = np.load(x[1]).astype(np.float32)
        obs  = np.load(x[2]).astype(np.float32)

        pred = pred[(3, 0, 1, 2), :, :]
    
        pred = preprocess(pred)
        obs  = preprocess(obs)

        MAE_24 = check_MAE(obs, pred)

        list_24.append(MAE_24)
    a = np.array(list_24)
    print(np.mean(a, axis=0))


    #np.save('../data/pangu_24_MAE_loss.npy', np.array(list_24))
    
        

if __name__ == '__main__':
    main()
