from netCDF4 import Dataset as nc
import matplotlib.pyplot as plt
import numpy as np
import ease_grid as eg
import time
import datetime
from  glob import glob
import pandas as pd
import csv
import pickle
# file = open('C:/users/kellen/programming/latlonborehole.csv')
# f = csv.reader(file)

# year = 2016
# files = [(int(i.split('\\')[1].split(',')[0]),int(i.split('\\')[1].split(',')[1])) for i in glob('C:/users/kellen/programming/Borehole data/[1-9]*.csv')]

#
# bores = []
# for c in files:
#     if (c[1] == 2016) or (c[1] == 2017) or (c[1] == 2018):
#         bores.append(c[0])
#
# bores = np.unique(bores)
#
# bore_dict = {}
# for i in f:
#     line1 = i
#     line2 = next(f)
#     line3 = (line1[0] + line2[0]).split()
#     station = int(line3[0])
#     lat = int(line3[1])+int(line3[3])/60
#     lon = int(line3[6]) + int(line3[8]) / 60
#     x,y = eg.v2_lonlat_to_colrow_coords(-lon,lat,'N9')
#     y = int(round(y))
#     x = int(round(x))
#     if (y>=2000) or (y<=0) or (x>=2000) or (x<=0):
#         continue
#
#     if station in bores:
#         if (y,x) not in bore_dict:
#             bore_dict[(y,x)] = [f'USDA{station}']
#         else:
#             bore_dict[(y,x)].append(f'USDA{station}')

with open('updateddict5.pickle', 'rb') as handle:
    bore_dict = pickle.load(handle)
count = 0
for coords in bore_dict:
    stations = bore_dict[coords]
    if stations[0][:4] != 'USDA':
        print('skipped',stations[0])
        continue
    for station in stations:
        df = pd.DataFrame()
        df['date'] = []
        df['temp'] = []
        for year in [2015,2016,2017,2018,2019,2020,2021]:
            try:
                if year < 2019:
                    data = pd.read_csv(f'C:/users/kellen/programming/Borehole data/{station[4:]},{year},18.csv', skiprows=4)
                else:
                    data = pd.read_csv(f'C:/users/kellen/programming/newbores/{station[4:]},{year},18.csv', skiprows=4)
            except:
                continue
            temp = pd.DataFrame()
            temp['date'] = []
            temp['temp'] = []
            if len(data['Date']) < 30:
                continue

            try:
            
                temp[['date', 'temp']] = data[['Date','STO.I-1:-2 (degC) ']]
            except:
                try:
                    temp[['date', 'temp']] = data[['Date', 'STO.I-1:0 (degC) ']]
                except:
                    print('skipping',station,f'on {year}')
                    continue
            df = pd.concat([df,temp])

        df.to_csv(f'boreholesE/{station}.csv',index=False)
print(count)
