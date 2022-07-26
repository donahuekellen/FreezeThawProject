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
from scipy.interpolate import NearestNDInterpolator as ind
# from scipy.interpolate import griddata as gr
import glob




year = 2016
ndvivars = nc('F:/data/NDVI/AVHRR-Land_v005-preliminary_AVH13C1_NOAA-19_20181231_c20190130113506.nc')
ndvi_data = nc(f'E:/data/gapfilledndvi{year}.nc')
mask = np.load('9K_NH_PROJ_LAND_MASK.npy')

lons = ndvivars.variables['longitude'][:].data
lats = ndvivars.variables['latitude'][:1800].data
lon = [j for i in lats for j in lons]
lat = [i for i in lats for j in lons]

x,y = eg.v2_lonlat_to_colrow_coords(lon,lat,'N9')
X = [j for i in range(2000) for j in range(2000)]
Y = [i for i in range(2000) for j in range(2000)]
coords = list(zip(x,y))
start = datetime.datetime(year,1,1)
rootgrp = nc(f"E:/data/ndviformatted{year}.nc", "w", format="NETCDF4")
t = rootgrp.createDimension("time", None)
freq = rootgrp.createDimension("freq", 1)
y = rootgrp.createDimension("y", 2000)
x = rootgrp.createDimension("x", 2000)
times = rootgrp.createVariable('time','i2',('time'))
times.units = f'days since {start.date()} 06:00:00'
tbs = rootgrp.createVariable('tb', 'f4', ('time', 'freq', 'y', 'x'),zlib=True,complevel=4,least_significant_digit = 2,fill_value=0)
tbs.frequency = 'ERA5 skin surface temperature 1'
tbs.units = 'K'


for k,i in enumerate(range(len(ndvi_data.variables['tb']))):
    print(k)
    era = ndvi_data.variables['tb'][i]
    t = ind(coords,era.flatten())
    out = t(X,Y).reshape(2000,2000)
    tempmask = np.isnan(out)
    out[tempmask|(mask == False)] = 0
    temp = np.ma.masked_array(out,tempmask|(mask == False))
    tbs[k] = temp
    times[k] = k