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

goodfiles = []
for file in glob('./GTNB/Boreholes/*.csv'):
    skip = False
    for i in re.split('-|_|\.',file):
        if i =='satellite':
            skip = True
            break
        if i =='Air':
            skip = True
            break
    if skip:
        continue
    goodfiles.append(file)

gooddfs = []
files= []
for file in goodfiles:
    skip = True
    df = pd.read_csv(file)
    for i in df.columns.values[1:]:
        if float(i) < .1:
            skip = False
    if skip:
        continue
    if int(df['Date/Depth'].loc[len(df)-1][:4]) < 2016:
        continue

    gooddfs.append(df)
    files.append(file)
finaldfs = []
finalfiles = []
for k,df in enumerate(gooddfs):
    newdf = pd.DataFrame()
    newdf['date'] = [i[:10] for i in df['Date/Depth'].values]
    curr = 100
    for i in df.columns.values[1:]:
        if float(i)>=0 and abs(float(i)-.05) < curr:
            curr = abs(float(i)-.05)
            newdf['temp'] = df[i].values

    try:
        ind = newdf[newdf['date'] == '2016-01-01'].index[0]
        newdf = newdf.iloc[ind:]
        newdf.reset_index(inplace=True,drop=True)
        if newdf.loc[0]['temp'] == -999:
            continue
    except:
        continue
    newdf['temp'][newdf['temp']==-999] = np.nan
    print(files[k].split('\\')[1].split('-')[1],curr)
    finalfiles.append(files[k].split('\\')[1].split('-Dataset')[0])
    finaldfs.append(newdf)
    name = files[k].split('\\')[1].split('-Dataset')[0]
    newdf.to_csv(f'boreholesE/{name}.csv', index=False)


with open('updateddict2.pickle', 'rb') as handle:
    dicts = pickle.load(handle)


for meta in finalfiles:
    with open(f'./GTNB/Boreholes/{meta}.metadata.txt', encoding="utf8") as f:
        lines = f.readlines()
        for i in lines:
            line = i.split(':')
            if line[0] == 'Longitude':
                lon = float(line[1].strip())
            elif line[0] == 'Latitude':
                lat = float(line[1].strip())
                break
        x, y = eg.v2_lonlat_to_colrow_coords(lon, lat, 'N9')
        y = int(round(y))
        x = int(round(x))
        if (y, x) not in dicts:
            dicts[(y, x)] = [meta]
        else:
            dicts[(y, x)].append(meta)
with open('updateddict3.pickle', 'wb') as handle:
    pickle.dump(dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
