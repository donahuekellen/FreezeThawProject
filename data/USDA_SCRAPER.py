# -*- coding: utf-8 -*-
"""
Created on Wed May 22 19:10:36 2019

@author: Kellen
"""

import requests
import time
from bs4 import BeautifulSoup as bs
import os
import csv

# file = open('C:/users/kellen/programming/latlonborehole.csv')
# g = csv.reader(file)
# year = 2019
# workingYear = 2019
# for i in g:
    # line1 = i
    # line2 = next(g)
    # line3 = (line1[0] + line2[0]).split()
    # sitenum = int(line3[0])
    # sitecheck = 'https://wcc.sc.egov.usda.gov/nwcc/site?sitenum='+str(sitenum)
    # r = requests.get(sitecheck)
    # html = r.content
    # soup = bs(html,features="html.parser")
    # table = soup.find('p')
    
    # try:
        # for row in table:
            # if row == 'Data and Information is not available for site #' + str(sitenum) + ', this may be a discontinued site.':
                # print (str(sitenum) + " invalid")
    # except:
        # try:
             # test = soup.find('select', attrs={'id' :  'report'})
             # for row in test.find('option', attrs={'value' :  '29'}):
                 # pass
             # print(str(sitenum) + " valid")
             # while year < 2022:
                 # name = str(sitenum) + "," + str(year) + ",06"
                 # f=open(name+".csv","w+")
                 # data = {
                         # 'intervalType' : 'View Historic',
                         # 'sitenum' : str(sitenum),
                         # 'report' : '29',
                         # 'timeseries' : 'Hour:06',
                         # 'format' : 'copy',
                         # 'interval' : 'MONTH',
                         # 'year' : str(year),
                         # 'month' : 'CY'
                         # }
                 # dataurl = 'https://wcc.sc.egov.usda.gov/nwcc/view'
                 # response = requests.post(url=dataurl,data = data)
                 # if len(response.text) > 460:
                     # f.write(response.text)
                     # if workingYear == 1993: workingYear = year
                     # f.close()
                 # else:
                     # f.close()
                     # os.remove(name+".csv")
                 # year +=1
                 # time.sleep(.2)
        # except:
           # print(str(sitenum) + " invalid")
        # time.sleep(1)
        # year = workingYear;
        
year = 2016
workingYear = 2016
sitenum = 2229
while sitenum < 2240:
    sitecheck = 'https://wcc.sc.egov.usda.gov/nwcc/site?sitenum='+str(sitenum)
    r = requests.get(sitecheck)
    html = r.content
    soup = bs(html,features="html.parser")
    table = soup.find('p')
    
    try:
        for row in table:
            if row == 'Data and Information is not available for site #' + str(sitenum) + ', this may be a discontinued site.':
                print (str(sitenum) + " invalid")
    except:
        try: 
             test = soup.find('select', attrs={'id' :  'report'})
             for row in test.find('option', attrs={'value' :  '29'}):
                 pass
             print(str(sitenum) + " valid")    
             while year < 2022:
                 name = str(sitenum) + "," + str(year) + ",06"
                 f=open(name+".csv","w+")     
                 data = {
                         'intervalType' : 'View Historic',
                         'sitenum' : str(sitenum),
                         'report' : '29',
                         'timeseries' : 'Hour:06',
                         'format' : 'copy',
                         'interval' : 'MONTH',
                         'year' : str(year),
                         'month' : 'CY'
                         }
                 dataurl = 'https://wcc.sc.egov.usda.gov/nwcc/view'
                 response = requests.post(url=dataurl,data = data)
                 if len(response.text) > 460:
                     f.write(response.text)
                     if workingYear == 1993: workingYear = year
                     f.close()
                 else:
                     f.close()
                     os.remove(name+".csv")
                 year +=1
                 time.sleep(.2)
        except:
           print(str(sitenum) + " invalid")
        time.sleep(1)
        year = workingYear;    

    year = 2014
    workingYear = 2014
    sitenum +=1
    if sitenum == 1300:
        sitenum = 2000
    time.sleep(.5)


