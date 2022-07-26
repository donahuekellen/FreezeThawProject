import requests
import time
from bs4 import BeautifulSoup as bs
import os
import glob
import pandas as pd
sitenums = []
os.chdir("C:/users/kellen/programming/Borehole Data")
for file in glob.glob("*,*"):
    sitenum = file.split(",")[0]
    if sitenum in sitenums:
        pass
    else:
        sitenums.append(sitenum)


df = pd.DataFrame(columns=['site','elev'])
for sitenum in sitenums:
    sitecheck = f'https://wcc.sc.egov.usda.gov/nwcc/site?sitenum={sitenum}'
    r = requests.get(sitecheck)
    html = r.content
    soup = bs(html,features="html.parser")
    text = soup.findAll(text=True)
    for k,t in enumerate(text):
        if t == 'Elevation:':
            s = pd.DataFrame([[int(sitenum), int(text[k + 1].split()[0]) * 0.3048]], columns=['site', 'elev'])
            df = pd.concat([df,s])
            break
    print(sitenum,int(text[k+1].split()[0])*0.3048)
    time.sleep(.2)
df.to_csv('elevborehole.csv')