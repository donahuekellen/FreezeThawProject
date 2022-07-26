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
import pickle

with open('./updateddict6.pickle', 'rb') as handle:
    dict = pickle.load(handle)

for folder in glob('C:/users/kellen/downloads/europe_boreholes/*/'):
    for borehole in glob(f'{folder}/*/'):
        for file in glob(f'{borehole}/*.stm'):

            if (file.split('_')[4] == 'ts' and float(file.split('_')[5]) >= 0 and float(file.split('_')[6]) <= .05) or (file.split('_')[5] == 'ts' and float(file.split('_')[6]) >= 0 and float(file.split('_')[7]) <= .05):
                # print(file)
                test = pd.read_csv(file, sep = '\s+',names=['date','time','cse','network','station','lat','lon','elev','depthfrom','depthto','value','ismnflag','providerflag'])
                lat = test['lat'][0]
                lon = test['lon'][0]
                station = test['station'][0]
                print(test['network'][0])
                x, y = eg.v2_lonlat_to_colrow_coords(lon, lat, 'N9')
                y = int(round(y))
                x = int(round(x))
                if (y, x) not in dict:
                    dict[(y, x)] = [f'{station}']
                else:
                    dict[(y, x)].append(f'{station}')
                
                df = pd.DataFrame()
                df['date'] = []
                df['temp'] = []
                dates = []
                temps = []
                for i in test.iterrows():
                    if i[1]['time'] == '06:00':
                        temps.append(i[1]['value'])
                        dates.append(i[1]['date'].replace('/','-'))
                
                df['date'] = dates
                df['temp'] = temps
                df.to_csv(f'boreholes_fixed/{station}.csv', index=False)

with open('updateddict6.pickle', 'wb') as handle:
    pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

