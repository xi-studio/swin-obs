import glob
import time

res = glob.glob('data/72/*/*.npz')

res.sort()


for x in res:
    print(x)
    name = x.split('/')[-1]
    print(name[:-4])

    r = time.strptime('2020-01-04T00', '%Y-%m-%dT%H')
    sname = time.strftime('%Y%m%d%H%M', r)
    print(sname)
    break

