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
    cat = np.mean(np.abs(pred - x))

    return (tem, mslp, u, v, cat)


def main():
    filename = '../data/meta/pangu_meta.npy'
    res = np.load(filename)

    list_1  = []
    list_6  = []
    list_24 = []

    for i, x in enumerate(res):
        print(i)
        obs = preprocess(np.load(x[0]))
        pred_1  = preprocess(np.load(x[1])[(3,0,1,2), :, :])
        pred_6  = preprocess(np.load(x[2])[(3,0,1,2), :, :])
        pred_24 = preprocess(np.load(x[3])[(3,0,1,2), :, :])
    
        MAE_1  = check_MAE(obs, pred_1) 
        MAE_6  = check_MAE(obs, pred_6)
        MAE_24 = check_MAE(obs, pred_24)

        list_1.append(MAE_1)
        list_6.append(MAE_6)
        list_24.append(MAE_24)
        break

    np.save('../data/MAE_loss.npy', np.array((list_1, list_6, list_24)))
    
        

if __name__ == '__main__':
    main()
