import torch
import os
import numpy as np
import onnx
import onnxruntime as ort
import xarray as xr
import dask
import numpy as np
from typing import OrderedDict
import yaml
import time


def load_vars(var_config_file):
    with open(var_config_file, "r") as f:
        var_config = yaml.load(f, Loader=yaml.Loader)

    return var_config

def main():

    ar_full_37_1h = xr.open_zarr(
        'gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2/'
    )
    print("Model surface dataset size {} TiB".format(ar_full_37_1h.nbytes/(1024**4)))

    variables = OrderedDict(load_vars('./pangu_config.yaml'))
    surface_data = ar_full_37_1h[variables['input']['surface']]
    high_data = ar_full_37_1h[variables['input']['high']].sel(level=variables['input']['levels'])
    sur  = surface_data.sel(time=slice('2020-01-01', '2021-12-31'))
    hig  = high_data.sel(time=slice('2020-01-01', '2021-12-31'))
    sur_dat = xr.concat([sur['mean_sea_level_pressure'],sur['10m_u_component_of_wind'],sur['10m_v_component_of_wind'], sur['2m_temperature']],"var")
    hig_dat = xr.concat([hig['geopotential'],hig['specific_humidity'],hig['temperature'],hig['u_component_of_wind'],hig['v_component_of_wind']],'var')

    res_sur  = sur_dat.transpose('time','var','latitude','longitude')
    res_high = hig_dat.transpose('time','var','level','latitude','longitude')


    for i, _ in enumerate(res_sur):
        now = str(res_sur[i]['time'].to_numpy())
        now = now.split(":")[0]

        sur = res_sur[i].to_numpy()
        high = res_high[i].to_numpy()

        sur = sur.astype(np.float16)
        high = sur.astype(np.float16)

        obs_sur = sur[:, 30*4:90*4+1, 70*4:140*4+1]
        obs_upper = high[:, :, 30*4:90*4+1, 70*4:140*4+1]
        np.savez('./pangu_2020_2021/%s.npy' % now, sur=obs_sur, upper=high)
        print('save', now)
        break



if __name__ == '__main__':
    main()
