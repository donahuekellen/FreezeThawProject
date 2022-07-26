from netCDF4 import Dataset as nc
import matplotlib.pyplot as plt
import numpy as np
import ease_grid as eg
import time
import datetime
from  glob import glob
import pandas as pd
import csv
import os
import re
import pickle

dic = {}
loc = 0
for file in glob('./globe/*.csv'):
    test = pd.read_csv(file,skiprows=[1])
    for i in test[' site_name'].unique():
        if i in dic:
            pass
        else:
            dic[i] = loc
            loc+=1

start = datetime.datetime(2016, 1, 1)
end = datetime.datetime(2022, 1, 1)
dates = [str((start + datetime.timedelta(days=i)).date()) for i in range((end - start).days)]
df = pd.DataFrame(index= np.arange(len(dic.keys())),columns=['name','lat','lon',*dates])
current = None
closest = None
bore = None
for file in glob('./globe/*.csv'):
    print('start of', file)
    test = pd.read_csv(file,skiprows=[1])
    for i in test.iterrows():

        if i[1]['soil temp sub days:depth level (cm)'] <=5 and (i[1][ ' measured_on'] != current or i[1][' site_name'] != bore):
            if i[1][' site_name'] != bore:
                df['name'][dic[i[1][' site_name']]] = i[1][' site_name']
                df['lat'][dic[i[1][' site_name']]] = i[1][' latitude']
                df['lon'][dic[i[1][' site_name']]] = i[1][' longitude']
                bore = i[1][' site_name']
            current = i[1][' measured_on']
            df[i[1]['soil temp sub days:measured at'][:10]][dic[i[1][' site_name']]] = i[1]['soil temp sub days:current temp (deg C)']
            closest = int(i[1]['soil temp sub days:measured at'][11:13])
        elif i[1]['soil temp sub days:depth level (cm)'] <=5 and abs(int(i[1]['soil temp sub days:measured at'][11:13])-18)<abs(closest-18):
                closest = int(i[1]['soil temp sub days:measured at'][11:13])
                df[i[1]['soil temp sub days:measured at'][:10]][dic[i[1][' site_name']]] = i[1]['soil temp sub days:current temp (deg C)']

temp = df.dropna(how='all')
temp = temp.reset_index()
temp = temp.drop('index',axis=1)

with open('updateddict4.pickle', 'rb') as handle:
    dicts = pickle.load(handle)
for row in temp.iterrows():
    newdf = pd.DataFrame()
    newdf['date'] = row[1].index[3:].values
    newdf['temp'] = row[1][3:].values
    if len(newdf.dropna())>30:
        name = row[1]['name'].replace('\'', " ").replace('\"', " ").replace('?', " ").replace(':', " ").replace('/',"").replace('\\',"")
        print(name)
        x, y = eg.v2_lonlat_to_colrow_coords(row[1]['lon'], row[1]['lat'], 'N9')
        y = int(round(y))
        x = int(round(x))
        if (y,x) not in dicts:
            dicts[(y,x)] = [name]
        else:
            if name not in dicts[(y,x)]:
                dicts[(y,x)].append(name)
        newdf.to_csv(f'boreholesE/{name}.csv', index=False)
