from netCDF4 import Dataset as nc
import numpy as np
import datetime

"""
Fills the currently missing data with the previous/next day's data and gives it a weight
depending on how far the day is from the date being filled.
Catches and ignores missing dates, mostly for handling the start and end of the current data record.
"""
def data_filler(fill,weights,fill_date,days,pol,pas):
    try:
        date = fill_date+datetime.timedelta(days)
        diff = date - datetime.datetime(date.year - 1, 12, 31)
        data = nc(f'H:/smap2/NSIDC-0738-EASE2_N09km-SMAP_LRM-{date.year}{diff.days:03}-1.4{pol}-{pas}-SIR-JPL-v2.0.nc').variables['TB'][0]
        data[data>350] = 0
        mask = (fill==0) * (data.mask == False)
        fill[mask] = data[mask]
        weights[mask] = abs(days)
    except:
        print(f'{date.date()} DNE')

"""
Fills the current date's missing data with a weighted average of the previous and next 5 days.
"""
def gap_fill(date,pol,pas):
    diff = date - datetime.datetime(date.year - 1, 12, 31)
    day = nc(f'H:/smap2/NSIDC-0738-EASE2_N09km-SMAP_LRM-{date.year}{diff.days:03}-1.4{pol}-{pas}-SIR-JPL-v2.0.nc').variables['TB'][0]
    fill = np.zeros((2,day.shape[0],day.shape[1]))
    day[day > 350] = 0
    fill[0] = day
    fill[1] = day
    weights = np.zeros_like(fill)
    weights[fill!=0] = .5
    for i in range(5):
        data_filler(fill[0], weights[0], date, -(i+1),pol,pas)
        data_filler(fill[1], weights[1], date, i+1,pol,pas)
    total = np.sum(weights,0)
    weights = 1-weights/total

    #necessary for fixing the weights of pixels that don't have data in the prev/next
    #5 days but do have data for the next/prev 5 days
    ones = weights==1
    zeros = weights==0
    weights[ones] = 0
    weights[zeros] = 1
    return np.average(fill,0,weights)





