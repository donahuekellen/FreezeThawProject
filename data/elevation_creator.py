from netCDF4 import Dataset as nc
import numpy as np
import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import datetime
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from glob import glob

with open('updateddict6.pickle', 'rb') as handle:
    dict = pickle.load(handle)
with open('elevdict.pickle', 'rb') as handle:
    elevdict = pickle.load(handle)

ismnelevs = pd.DataFrame(columns=['site','elev'])

for folder in glob('C:/users/kellen/downloads/europe_boreholes/*/'):
    for borehole in glob(f'{folder}/*/'):
        for file in glob(f'{borehole}/*.stm'):

            if (file.split('_')[4] == 'ts' and float(file.split('_')[5]) >= 0 and float(file.split('_')[6]) <= .05) or (file.split('_')[5] == 'ts' and float(file.split('_')[6]) >= 0 and float(file.split('_')[7]) <= .05):
                # print(file)
                test = pd.read_csv(file, sep = '\s+',names=['date','time','cse','network','station','lat','lon','elev','depthfrom','depthto','value','ismnflag','providerflag'])
                lat = test['lat'][0]
                lon = test['lon'][0]
                elev = test['elev'][0]
                station = test['station'][0]
                print(station,elev)
                elevdict[station] = elev
                break




for file in glob('./globe/*.csv'):
    test = pd.read_csv(file,skiprows=[1])
    temp = np.array([i.replace('\'', " ").replace('\"', " ").replace('?', " ").replace(':', " ").replace('/',"").replace('\\',"") for i in test[' site_name'].unique()])
    temp2 = np.array([i for i in test[' site_name'].unique()])
    for i in dict.values():
        for j in i:
            if j in temp:
                tt = temp2[np.argwhere(np.array(temp) == j)[0][0]]
                e =  test[test[' site_name'] == tt].iloc[0][' elevation']
                print(j,e)
                if j not in elevdict:
                    elevdict[j] = e

temp = np.array([file.split('\\')[1].split('-Dataset')[0] for file in glob('./GTNB/Boreholes/*.csv')])
for i in dict.values():
    for j in i:
        if j in temp:
            t = open(f'./GTNB/Boreholes/{j}.metadata.txt', encoding="mbcs")
            text = t.readlines()
            elev = float(text[12][11:].strip())
            elevdict[j] = elev
