from netCDF4 import Dataset as nc
import matplotlib.pyplot as plt
import numpy as np
import ease_grid as eg
import matplotlib.cm as cm
import time
import datetime
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.ndimage import convolve as bin
from pyproj import Proj, transform
from scipy.interpolate import LinearNDInterpolator as ind

era_data = nc('C:/users/kellen/downloads/adaptor.mars.internal-1633023466.8052747-11247-16-aeefb4a4-cbfa-4f64-b5a6-a636759f5217.nc')
mask = np.load('9K_NH_PROJ_LAND_MASK.npy')

lons = era_data.variables['longitude'][:].data
lats = era_data.variables['latitude'][:360].data
lon = [j for i in lats for j in lons]
lat = [i for i in lats for j in lons]

x,y = eg.v2_lonlat_to_colrow_coords(lon,lat,'N9')
X = [j for i in range(2000) for j in range(2000)]
Y = [i for i in range(2000) for j in range(2000)]
coords = list(zip(x,y))
year = 2020
start = datetime.datetime(year,1,1)
rootgrp = nc(f"E:/data/era{year}E.nc", "w", format="NETCDF4")
t2 = rootgrp.createDimension("time", None)
freq = rootgrp.createDimension("freq", 1)
y = rootgrp.createDimension("y", 2000)
x = rootgrp.createDimension("x", 2000)
times = rootgrp.createVariable('time','i2',('time'))
times.units = f'days since {start.date()} 06:00:00'
tbs = rootgrp.createVariable('tb', 'f4', ('time', 'freq', 'y', 'x'),zlib=True,complevel=4,least_significant_digit = 2,fill_value=0)
tbs.frequency = 'ERA5 skin surface temperature 1'
tbs.units = 'K'

temp = era_data.variables['stl1'][0][:360]
t = ind(coords,temp.flatten())
for k,i in enumerate(range(len(era_data.variables['stl1']))[1::2]):
    print(k)
    era = era_data.variables['stl1'][i][:360]
    t.values = era.flatten().data.reshape(-1,1)
    out = t(X,Y).reshape(2000,2000)
    tempmask = np.isnan(out)
    out[tempmask|(mask == False)] = 2
    temp = np.ma.masked_array(out,tempmask|(mask == False))
    tbs[k] = temp
    times[k] = k
rootgrp.close()