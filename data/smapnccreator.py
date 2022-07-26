from netCDF4 import Dataset as nc
import matplotlib.pyplot as plt
import numpy as np
import ease_grid as eg
import matplotlib.cm as cm
import time
from gap_fill_smap import *
import datetime

s = time.time()
test = nc('C:/users/kellen/downloads/NSIDC-0738-EASE2_N09km-SMAP_LRM-2019094-1.4V-E-SIR-JPL-v1.0.nc')
X = test.variables['x'][:].data
Y = test.variables['y'][:].data
X_METERS = [j for i in Y for j in X]
Y_METERS = [i for i in Y for j in X]
mask = np.load('9K_NH_PROJ_LAND_MASK.npy')
alon, alat = eg.v2_meters_to_lonlat(X_METERS,Y_METERS,'N9')
alon = np.reshape(alon,(2000,2000))
alat = np.reshape(alat,(2000,2000))
test.close()


year = 2020
pas = 'E'
if year == 2015:
        start = datetime.datetime(2014,12,31)+datetime.timedelta(90)
        end = datetime.datetime(2016,1,1)
elif year == 2021:
        start = datetime.datetime(2021, 1, 1)
        end = datetime.datetime(2020, 12, 31) + datetime.timedelta(120)
else:
        start = datetime.datetime(year,1,1)
        end = datetime.datetime(year+1, 1, 1)
dates = [start + datetime.timedelta(i) for i in range((end-start).days)]
rootgrp = nc(f"E:/data/SMAP{year}{pas}.nc", "w", format="NETCDF4")
t = rootgrp.createDimension("time", None)
freq = rootgrp.createDimension("freq", 2)
y = rootgrp.createDimension("y", 2000)
x = rootgrp.createDimension("x", 2000)
times = rootgrp.createVariable('time','i2',('time'))
times.units = f'days since {start.date()}'
tbs = rootgrp.createVariable('tb', 'f4', ('time', 'freq', 'y', 'x'),zlib=True,complevel=4,least_significant_digit = 2,fill_value=0)
lats = rootgrp.createVariable('lat', 'f4', ('x','y'))
lons = rootgrp.createVariable('lon', 'f4', ('x','y'))
print('here')
lats[:] = alat[:]
lons[:] = alon[:]
tbs.frequency = 'Vertical 1.4GHz, Horizontal 1.4GHz'
tbs.units = 'K'

for k,d in enumerate(dates):
        print(f'{d.date()}')
        filled = gap_fill(d,'V',pas)
        filled[np.isnan(filled)] = 0
        filled2 = gap_fill(d,'H',pas)
        filled2[np.isnan(filled2)] = 0
        temp = np.ma.masked_array([filled,filled2],[(filled==0)|(mask==False),(filled2==0)|(mask==False)])
        tbs[k] = temp
        times[k] = k
rootgrp.close()
print(time.time()-s)