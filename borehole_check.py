from netCDF4 import Dataset as nc
import matplotlib.pyplot as plt
import numpy as np
import ease_grid as eg
import time
import datetime
import torch
from  glob import glob
import pandas as pd
import csv
import pickle

class borehole_check():
    def __init__(self,y_min = None,y_max = None,x_min = None,x_max = None,time='M'):

        self.y_min = None
        self.y_max = None
        self.x_min = None
        self.x_max = None
        self.boreholes = {}
        self.prefix = None
        with open('./data/updateddict6.pickle', 'rb') as handle:
            self.dict = pickle.load(handle)
        if y_min == None:
            y_min = 0
        if y_max == None:
            y_max = 2000
        if x_min == None:
            x_min = 0
        if x_max == None:
            x_max = 2000
        if time == 'M':
            prefix = './data/boreholes_fixed/'
        else:
            prefix = './data/boreholesE/'

        self.y_min = y_min
        self.y_max = y_max
        self.x_min = x_min
        self.x_max = x_max

        for y in range(y_min,y_max):
            for x in range(x_min,x_max):
                if (y,x) in self.dict:
                    for hole in self.dict[(y,x)]:
                        try:
                            temp = pd.read_csv(prefix+f'{hole}.csv',sep='[;,]',engine='python')
                            # temp['temp'] = temp['temp'] > 0
                            # temp['temp'] = temp['temp'].astype(int)
                            if (y, x) not in self.boreholes:
                                self.boreholes[(y, x)] = [temp]
                            else:
                                self.boreholes[(y, x)].append(temp)
                        except:
                            print('missed',hole)#self.dict[(y,x)])



    def create_borehole_arrays(self,dates):
        arrays = []
        masks = []
        for date in dates:
            date = str(date.date())
            temp = np.zeros((1,self.y_max-self.y_min,self.x_max-self.x_min))
            mask = np.ones((1,self.y_max-self.y_min,self.x_max-self.x_min)).astype(bool)

            for borehole in self.boreholes:
                data_group = self.boreholes[borehole]
                ks = []
                for data in data_group:
                    if date in data['date'].values:
                            tempval = data[data['date'] == date]['temp'].item()
                            if tempval > -90:
                                ks.append(tempval)
                            else:
                                ks.append(np.nan)
                k = np.nanmean(ks)
                if not np.isnan(k):
                    temp[0,borehole[0]-self.y_min,borehole[1]-self.x_min] = int(k  >0)
                    mask[0,borehole[0] - self.y_min, borehole[1] - self.x_min] = False
            arrays.append(temp)
            masks.append(mask)
        return arrays,masks


    def get_borehole_accuracy(self,prediction,date):
        scores = []

        for borehole in self.boreholes:
            data = self.boreholes[borehole]
            if date in data['date'].values:
                scores.append(int(prediction[borehole[0]-self.y_min,borehole[1]-self.x_min] == data[data['date'] == date]['temp'].item()))

        if len(scores) != 0:
            return np.sum(scores)/len(scores)
        else:
            return np.nan
