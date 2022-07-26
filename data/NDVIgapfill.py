from netCDF4 import Dataset as nc
import matplotlib.pyplot as plt
import time
import numpy as np
import glob
import datetime
import os



files = glob.glob('F:/data/NDVI/*.nc')

dates = [i.split('_')[-2] for i in files]

dic = {}
for i in range(len(dates)):
    dic[dates[i]] = files[i]

mask = np.load('ndvimasktest.npy')[:1800]
datesdict = {}
year = 2015
start = datetime.datetime(year, 1, 1)
end = datetime.datetime(year+1, 1, 1)
datesdict[2015] =  [start + datetime.timedelta(days=i) for i in range((end - start).days)]
year = 2016
start = datetime.datetime(year, 1, 1)
end = datetime.datetime(year+1, 1, 1)
datesdict[2016] =  [start + datetime.timedelta(days=i) for i in range((end - start).days)]
year = 2017
start = datetime.datetime(year, 1, 1)
end = datetime.datetime(year+1, 1, 1)
datesdict[2017] =  [start + datetime.timedelta(days=i) for i in range((end - start).days)]
year = 2018
start = datetime.datetime(year, 1, 1)
end = datetime.datetime(year+1, 1, 1)
datesdict[2018] =  [start + datetime.timedelta(days=i) for i in range((end - start).days)]
year = 2019
start = datetime.datetime(year, 1, 1)
end = datetime.datetime(year+1, 1, 1)
datesdict[2019] =  [start + datetime.timedelta(days=i) for i in range((end - start).days)]
ndvidict = {}
ndvidict[2015] = nc('E:/data/ndvi2015.nc').variables['tb']
ndvidict[2016] = nc('E:/data/ndvi2016.nc').variables['tb']
ndvidict[2017] = nc('E:/data/ndvi2017.nc').variables['tb']
ndvidict[2018] = nc('E:/data/ndvi2018.nc').variables['tb']
ndvidict[2019] = nc('E:/data/ndvi2018.nc').variables['tb']


year = 2016
start = datetime.datetime(year, 1, 1)
test = nc(dic[f'20170101'])
rootgrp = nc(f"E:/data/gapfilledndvi{year}.nc", "w", format="NETCDF4")
t = rootgrp.createDimension("time", None)
freq = rootgrp.createDimension("freq", 1)
lat = rootgrp.createDimension("lat", 1800)
lon = rootgrp.createDimension("lon", 7200)
lon = test.variables['longitude']
lat = test.variables['latitude']
times = rootgrp.createVariable('time','i2',('time'))
times.units = f'days since {start.date()}'
tbs = rootgrp.createVariable('tb', 'f4', ('time', 'lat', 'lon'),zlib=True,complevel=4,least_significant_digit = 2,fill_value=0)
tbs.frequency = 'ERA5 skin surface temperature 1'
tbs.units = 'K'
test.close()



prevdata = []
prevqueuehead = 0
for i in range(100):
    temp = ndvidict[year-1][-1-i].data
    prevdata.append(temp)
nextdata = []
nextqueuehead = 0
for i in range(100):
    temp = ndvidict[year][1+i].data
    nextdata.append(temp)
l = len(prevdata)

for k,day in enumerate(datesdict[year]):
    s = time.time()
    curr = ndvidict[day.year][(day-datetime.datetime(day.year,1,1)).days].data
    missing = (curr < -1) * (mask == False)
    backward = curr.copy()
    bmissing = missing.copy()
    forward = curr.copy()
    fmissing = missing.copy()


    for i in range(l):
        queue = (prevqueuehead+i)%l
        temp = prevdata[queue]
        tempmask = (bmissing)*(temp>=-1)
        backward[tempmask] = temp[tempmask]
        bmissing = (backward<-1)*(mask==False)
        if np.all(backward[bmissing]>=-1):
            break
    prevdata[prevqueuehead] = curr
    prevqueuehead = (prevqueuehead-1)%l
    backward[backward<-1] = np.nan
    for i in range(l):
        queue = (nextqueuehead+i)%l
        temp = nextdata[queue]
        tempmask = (fmissing)*(temp>=-1)
        forward[tempmask] = temp[tempmask]
        fmissing = (forward<-1)*(mask==False)
        if np.all(forward[fmissing]>=-1):
            break
    newday = day+datetime.timedelta(100)
    nextdata[nextqueuehead] = ndvidict[newday.year][(newday-datetime.datetime(newday.year,1,1)).days].data
    nextqueuehead = (nextqueuehead+1)%l
    forward[forward < -1] = np.nan

    curr[missing] = np.nanmean([backward,forward],0)[missing]
    curr[np.isnan(curr)] = -9.999e3
    tbs[k] = np.ma.masked_array(curr,curr<-1)
    print(k,time.time()-s)






# year = 2015
# start = datetime.datetime(year, 6, 1)
# end = datetime.datetime(year+1, 1, 1)
# train_dates =  [start + datetime.timedelta(days=i) for i in range((end - start).days)]
#
# test = nc(dic[f'{start.year}{start.month:02}{start.day:02}'])
#
# rootgrp = nc(f"E:/data/ndvi{year}.nc", "w", format="NETCDF4")
# t = rootgrp.createDimension("time", None)
# freq = rootgrp.createDimension("freq", 1)
# lat = rootgrp.createDimension("lat", 1800)
# lon = rootgrp.createDimension("lon", 7200)
# lon = test.variables['longitude']
# lat = test.variables['latitude']
# times = rootgrp.createVariable('time','i2',('time'))
# times.units = f'days since {start.date()}'
# tbs = rootgrp.createVariable('tb', 'f4', ('time', 'lat', 'lon'),zlib=True,complevel=4,least_significant_digit = 2,fill_value=0)
# tbs.frequency = 'ERA5 skin surface temperature 1'
# tbs.units = 'K'
# test.close()
# for k,start in enumerate(train_dates):
#     print(k)
#     test = nc(dic[f'{start.year}{start.month:02}{start.day:02}'])
#     tbs[k] = main = test.variables['NDVI'][0].data[:1800]
# rootgrp.close()