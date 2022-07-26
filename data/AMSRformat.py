from netCDF4 import Dataset as nc
import matplotlib.pyplot as plt
import numpy as np
import ease_grid as eg
import matplotlib.cm as cm
import h5py as h
import time
import datetime
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.ndimage import convolve as bin
from pyproj import Proj, transform
from scipy.interpolate import NearestNDInterpolator as ind
# from scipy.interpolate import griddata as gr
import glob


test = nc('E:/data/AMSR2014A.nc')

lats = test.variables['lat'][:900].data.copy()
lons = test.variables['lon'][:].data.copy()
# year = 2016
# ndvivars = nc('F:/data/NDVI/AVHRR-Land_v005-preliminary_AVH13C1_NOAA-19_20181231_c20190130113506.nc')
# ndvi_data = nc(f'E:/data/gapfilledndvi{year}.nc')
mask = np.load('9K_NH_PROJ_LAND_MASK.npy')


# lons = ndvivars.variables['longitude'][:].data
# lats = ndvivars.variables['latitude'][:1800].data
lon = [j for i in lats for j in lons]
lat = [i for i in lats for j in lons]

x,y = eg.v2_lonlat_to_colrow_coords(lon,lat,'N9')
X = [j for i in range(2000) for j in range(2000)]
Y = [i for i in range(2000) for j in range(2000)]
coords = list(zip(x,y))
tempfile = h.File(f'H://AMSR2/ftp.gportal.jaxa.jp/standard/GCOM-W/GCOM-W.AMSR2/L3.TB18GHz_10/2/2016/03/GW1AM2_20160301_01D_EQMD_L3SGT18HA2220220.h5')
h36 = np.array(tempfile[f'Brightness Temperature (H)']).astype(float)
h36 = h36[:900]
t = ind(coords,h36.flatten())
for year in [2016,2017,2018,2019,2020]:

    start = datetime.datetime(year, 1, 1)
    end = datetime.datetime(year+1, 1, 1)
    dates = [start + datetime.timedelta(days=i) for i in range((end - start).days)]
    for fre in ['18','36']:
            rootgrp = nc(f"E:/data/AMSR{fre}gapfilled{year}HE.nc", "w", format="NETCDF4")
            tm = rootgrp.createDimension("time", None)
            freq = rootgrp.createDimension("freq", 1)
            y = rootgrp.createDimension("y", 2000)
            x = rootgrp.createDimension("x", 2000)
            times = rootgrp.createVariable('time','i2',('time'))
            times.units = f'days since {start.date()} 18:00:00'
            tbs = rootgrp.createVariable('tb', 'f4', ('time', 'freq', 'y', 'x'),zlib=True,complevel=4,least_significant_digit = 2,fill_value=0)
            tbs.frequency = 'ERA5 skin surface temperature 1'
            tbs.units = 'K'

            rootgrp2 = nc(f"E:/data/AMSR{fre}gapfilled{year}VE.nc", "w", format="NETCDF4")
            t2 = rootgrp2.createDimension("time", None)
            freq2 = rootgrp2.createDimension("freq", 1)
            y2 = rootgrp2.createDimension("y", 2000)
            x2 = rootgrp2.createDimension("x", 2000)
            times2 = rootgrp2.createVariable('time', 'i2', ('time'))
            times2.units = f'days since {start.date()} 06:00:00'
            tbs2 = rootgrp2.createVariable('tb', 'f4', ('time', 'freq', 'y', 'x'), zlib=True, complevel=4,
                                         least_significant_digit=2, fill_value=0)
            tbs2.frequency = 'ERA5 skin surface temperature 1'
            tbs2.units = 'K'


            for k,date in enumerate(dates):
                # date = start + datetime.timedelta(days=k)
                print(k)
                currfile = h.File(f'H://AMSR2/ftp.gportal.jaxa.jp/standard/GCOM-W/GCOM-W.AMSR2/L3.TB{fre}GHz_10/2/{date.year}/{date.month:02}/GW1AM2_{date.year}{date.month:02}{date.day:02}_01D_EQMA_L3SGT{fre}HA2220220.h5')
                h36 = np.array(currfile[f'Brightness Temperature (H)']).astype(float)
                h36 = h36[:900]

                v36 = np.array(currfile[f'Brightness Temperature (V)']).astype(float)
                v36 = v36[:900]

                missing = (h36 == 65534) #* (mask == False)
                backward = h36.copy()
                bmissing = missing.copy()
                forward = h36.copy()
                fmissing = missing.copy()

                missing2 = (v36 == 65534)  # * (mask == False)
                backward2 = v36.copy()
                bmissing2 = missing2.copy()
                forward2 = v36.copy()
                fmissing2 = missing2.copy()
                # s = time.time()
                for j in range(3):
                    prevdate = date + datetime.timedelta(days=(-(j+1)))
                    prevfile = h.File(f'H://AMSR2/ftp.gportal.jaxa.jp/standard/GCOM-W/GCOM-W.AMSR2/L3.TB{fre}GHz_10/2/{prevdate.year}/{prevdate.month:02}/GW1AM2_{prevdate.year}{prevdate.month:02}{prevdate.day:02}_01D_EQMA_L3SGT{fre}HA2220220.h5')
                    prev = np.array(prevfile[f'Brightness Temperature (H)']).astype(float)
                    prev = prev[:900]
                    backward[bmissing] = prev[bmissing]
                    bmissing = (backward  == 65534) #* (mask == False)

                    prev2 = np.array(prevfile[f'Brightness Temperature (V)']).astype(float)
                    prev2 = prev2[:900]
                    backward2[bmissing2] = prev2[bmissing2]
                    bmissing2 = (backward2 == 65534)  # * (mask == False)

                for j in range(3):
                    prevdate = date + datetime.timedelta(days=(j+1))
                    prevfile = h.File(
                        f'H://AMSR2/ftp.gportal.jaxa.jp/standard/GCOM-W/GCOM-W.AMSR2/L3.TB{fre}GHz_10/2/{prevdate.year}/{prevdate.month:02}/GW1AM2_{prevdate.year}{prevdate.month:02}{prevdate.day:02}_01D_EQMA_L3SGT{fre}HA2220220.h5')
                    prev = np.array(prevfile[f'Brightness Temperature (H)']).astype(float)
                    prev = prev[:900]
                    forward[fmissing] = prev[fmissing]
                    fmissing = (forward == 65534)# * (mask == False)

                    prev2 = np.array(prevfile[f'Brightness Temperature (V)']).astype(float)
                    prev2 = prev2[:900]
                    forward2[fmissing2] = prev2[fmissing2]
                    fmissing2 = (forward2 == 65534)  # * (mask == False)

                h36[missing] = np.nanmean([backward, forward], 0)[missing]

                v36[missing2] = np.nanmean([backward2, forward2], 0)[missing2]
                # print("fill time", time.time() - s)
                h36/= 100
                h36[h36>400] = 0
                h36[h36<0] = 0

                v36 /= 100
                v36[v36 > 400] = 0
                v36[v36 < 0] = 0

                t.values =  h36.flatten()
                out = t(X, Y).reshape(2000, 2000)
                tempmask = np.isnan(out)
                out[tempmask | (mask == False)] = 0
                temp = np.ma.masked_array(out, tempmask | (mask == False))
                tbs[k] = temp
                times[k] = k

                t.values = v36.flatten()
                out = t(X, Y).reshape(2000, 2000)
                tempmask = np.isnan(out)
                out[tempmask | (mask == False)] = 0
                temp = np.ma.masked_array(out, tempmask | (mask == False))

                tbs2[k] = temp
                times2[k] = k
            rootgrp.close()
            rootgrp2.close()