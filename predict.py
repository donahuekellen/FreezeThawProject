import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
#import matplotlib.pyplot as plt
import datetime
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#import matplotlib as mpl
from torch.utils.data import Dataset, DataLoader
# import h5py as h
#import ease_grid as eg
from netCDF4 import Dataset as nc
from model import UNet
#from borehole_check import borehole_check as bc
from sklearn.metrics import matthews_corrcoef as mc
#import matplotlib.cm as cm
import pickle
class custdata(Dataset):
    def __init__(self,years,folders = ['alaska'],features=[1,2,3,4,5,6,7,8],val=False,test=False,sub=0,time = 'M'):
        self.data = []
        self.labels = []
        self.boreholes = []
        self.years = years
        self.val = val
        self.features = features
        
        elev = np.array([np.load('/nobackup/kadonahu/smap/data/elevationnorthern.npy')]).astype(float)#[:,675:925,350:600]
        elev = (elev-elev.min())/(elev.max()-elev.min())
        with open('/nobackup/kadonahu/smap/data/lats.pickle','rb') as handle:
            lats = pickle.load(handle)
        subsections = [[0,2000,0,2000],[0,2000,1000,2000],[0,1000,1000,2000],[1000,2000,1000,2000]]
        subsection = subsections[sub]
        for year in years:
            if time =='M':
               smap = nc(f'/nobackup/kadonahu/smap/data/SMAP{year}M.nc')
               a18H = nc(f'/nobackup/kadonahu/smap/data/AMSR18gapfilled{year}H.nc')
               a18V = nc(f'/nobackup/kadonahu/smap/data/AMSR18gapfilled{year}V.nc')
               a36V = nc(f'/nobackup/kadonahu/smap/data/AMSR36gapfilled{year}V.nc')
               a36H = nc(f'/nobackup/kadonahu/smap/data/AMSR36gapfilled{year}H.nc')
               bores = nc(f'/nobackup/kadonahu/smap/data/bore{year}M.nc')
               era = nc(f'/nobackup/kadonahu/smap/data/era{year}M.nc')
            else:
               smap = nc(f'/nobackup/kadonahu/smap/data/SMAP{year}E.nc')
               a18H = nc(f'/nobackup/kadonahu/smap/data/AMSR18gapfilled{year}HE.nc')
               a18V = nc(f'/nobackup/kadonahu/smap/data/AMSR18gapfilled{year}VE.nc')
               a36V = nc(f'/nobackup/kadonahu/smap/data/AMSR36gapfilled{year}VE.nc')
               a36H = nc(f'/nobackup/kadonahu/smap/data/AMSR36gapfilled{year}HE.nc')
               bores = nc(f'/nobackup/kadonahu/smap/data/bore{year}E.nc')
               era = nc(f'/nobackup/kadonahu/smap/data/era{year}E.nc')
            start = datetime.datetime(year, 1, 1)
            end = datetime.datetime(year + 1, 1, 1)
            dates = [start + datetime.timedelta(days=i) for i in range((end - start).days)]
            

            length = 1
            times = 1
            if val:
                length = 3
                times = 1
                
            smapd = smap.variables['tb'][0]/350
            smapprev = smap.variables['tb'][0]/350

            v36 = a36V.variables['tb'][0]
            v36prev = a36V.variables['tb'][0]
            h36 = a36H.variables['tb'][0]
            h36prev = a36H.variables['tb'][0]
            v18 = a18V.variables['tb'][0]
            v18prev = a18V.variables['tb'][0]
            h18 = a18H.variables['tb'][0]
            h18prev = a18H.variables['tb'][0]
            
            v36[v36>350] = 0
            h36[h36>350] = 0
            v18[v18>350] = 0
            h18[h18>350] = 0
            
            v36/=350
            h36/=350
            v18/=350
            h18/=350
            
            v36prev[v36prev>350] = 0
            h36prev[h36prev>350] = 0
            v18prev[v18prev>350] = 0
            h18prev[h18prev>350] = 0
            
            v36prev/=350
            h36prev/=350
            v18prev/=350
            h18prev/=350
            
            data = np.concatenate([smapd,smapprev,lats,elev,v36,v36prev,h36,h36prev,v18,v18prev,h18,h18prev],0)
            data = data[features]
            label = era.variables['tb'][0]
            tempmask = label == 0
            label = (label > 273).astype(int)
            label[tempmask] = 2
            bore = bores.variables['tb'][0]
            self.data.append(torch.tensor(data).float())
            self.labels.append(torch.tensor(label).float())
            self.boreholes.append(torch.tensor(bore).unsqueeze(0).float())
            
            if test:
                for i in range(1,int(len(dates) / length)):
                    i *= length
                    
                    smapd = smap.variables['tb'][i]/350
                    smapprev = smap.variables['tb'][i-1]/350
                    
                    v36 = a36V.variables['tb'][i]
                    v36prev = a36V.variables['tb'][i-1]
                    h36 = a36H.variables['tb'][i]
                    h36prev = a36H.variables['tb'][i-1]
                    v18 = a18V.variables['tb'][i]
                    v18prev = a18V.variables['tb'][i-1]
                    h18 = a18H.variables['tb'][i]
                    h18prev = a18H.variables['tb'][i-1]
                    
                    v36[v36>350] = 0
                    h36[h36>350] = 0 
                    v18[v18>350] = 0 
                    h18[h18>350] = 0 
                    
                    v36/=350
                    h36/=350
                    v18/=350
                    h18/=350
                    
                    v36prev[v36prev>350] = 0
                    h36prev[h36prev>350] = 0
                    v18prev[v18prev>350] = 0
                    h18prev[h18prev>350] = 0
                    
                    v36prev/=350
                    h36prev/=350
                    v18prev/=350
                    h18prev/=350

                    data = np.concatenate([smapd,smapprev,lats,elev,v36,v36prev,h36,h36prev,v18,v18prev,h18,h18prev],0)
                    data = data[features]
                    label = era.variables['tb'][i]
                    tempmask = label == 0
                    label = (label > 273).astype(int)
                    label[tempmask] = 2
                    bore = bores.variables['tb'][i]
                    self.data.append(torch.tensor(data).float())
                    self.labels.append(torch.tensor(label).float())
                    self.boreholes.append(torch.tensor(bore).unsqueeze(0).float())
                        
               
            else:
                for i in range(int(len(dates) / length)):
                    i *= length
                    
                    smapd = smap.variables['tb'][i][:,subsection[0]:subsection[1],subsection[2]:subsection[3]]/350
                    smapprev = smap.variables['tb'][i]/350
                    v36 = a36V.variables['tb'][i][:,subsection[0]:subsection[1],subsection[2]:subsection[3]]
                    h36 = a36H.variables['tb'][i][:,subsection[0]:subsection[1],subsection[2]:subsection[3]]
                    v18 = a18V.variables['tb'][i][:,subsection[0]:subsection[1],subsection[2]:subsection[3]]
                    h18 = a18H.variables['tb'][i][:,subsection[0]:subsection[1],subsection[2]:subsection[3]]
                    
                    v36prev = a36V.variables['tb'][i-1]
                    h36prev = a36H.variables['tb'][i-1]
                    v18prev = a18V.variables['tb'][i-1]
                    h18prev = a18H.variables['tb'][i-1]
                    
                    v36[v36>350] = 0 
                    h36[h36>350] = 0
                    v18[v18>350] = 0
                    h18[h18>350] = 0
                    
                    v36/=350
                    h36/=350
                    v18/=350
                    h18/=350
                    
                    v36prev[v36prev>350] = 0
                    h36prev[h36prev>350] = 0
                    v18prev[v18prev>350] = 0
                    h18prev[h18prev>350] = 0
                    
                    v36prev/=350
                    h36prev/=350
                    v18prev/=350
                    h18prev/=350
                    
                    data = np.concatenate([smapd,smapprev,lats,elev,v36,v36prev,h36,h36prev,v18,v18prev,h18,h18prev],0)
                    data = data[features]
                    label = era.variables['tb'][i][:,subsection[0]:subsection[1],subsection[2]:subsection[3]]
                    tempmask = label == 0
                    label = (label > 273).astype(int)
                    label[tempmask] = 2
                    bore = bores.variables['tb'][i][subsection[0]:subsection[1],subsection[2]:subsection[3]]
                    self.data.append(torch.tensor(data).float())
                    self.labels.append(torch.tensor(label).float())
                    self.boreholes.append(torch.tensor(bore).unsqueeze(0).float())
                        


        self.data_len = len(self.data)
    def __getitem__(self, i):
        return self.labels[i].cuda(),self.data[i].cuda(),self.boreholes[i].cuda()

    def __len__(self):
        return self.data_len

    def shuffle(self,folders = ['alaska']):
        self.data = []
        self.labels = []
        self.boreholes = []
        for year in self.years:
            start = datetime.datetime(year, 1, 1)
            end = datetime.datetime(year + 1, 1, 1)
            dates = [start + datetime.timedelta(days=i) for i in range((end - start).days)]
            dates = dates

            length = 1
            times = 1
            if self.val:
                length = 10
                times = 1

            for i in range(0,int(len(dates) / length),2):
            
                i *= length
                
                for folder in folders:
                    data = np.load(f'E:/data/{folder}/data/{year}/{i}.npy')[self.features]
                    label = np.load(f'E:/data/{folder}/labels/{year}/{i}.npy')
                    bore = np.load(f'E:/data/{folder}/bores/{year}/{i}.npy')

                    for j in range(times):

                        k = np.random.randint(0, self.tilespace)
                        p = np.random.randint(0, self.tilespace)
                        for t in range(4):
                            tempd = np.rot90(data[:, k:k + int(175-self.tilespace), p:p + int(175-self.tilespace)], k=t, axes=(1, 2)).copy()
                            templ = np.rot90(label[:, k:k + int(175-self.tilespace), p:p + int(175-self.tilespace)], k=t, axes=(1, 2)).copy()
                            tempk = np.rot90(bore[:, k:k + int(175-self.tilespace), p:p + int(175-self.tilespace)], k=t, axes=(1, 2)).copy()
                            self.data.append(torch.tensor(tempd).float())
                            self.labels.append(torch.tensor(templ))
                            self.boreholes.append(torch.tensor(tempk))
        self.data_len = len(self.data)




def changelr(optimizer,divisor):
    for g in optimizer.param_groups:
        g['lr'] = g['lr']/divisor

def local_variation_loss(data, loss_func=nn.L1Loss()):
    """Compute the local variation around each pixel.

    This loss discourages high frequency noise on borders.
    """
    # Compute vertical variation
    loss = loss_func(data[..., 1:, :], data[..., :-1, :])
    # Compute horizontal variation
    loss += loss_func(data[..., :, 1:], data[..., :, :-1])
    return loss

def validate(model):
    with torch.no_grad():
        model.eval()
        predictedlist = []
        scores = []
        borescores = []
        borescore = 0
        boretotal = 0

        for k, (labels, data,bore) in enumerate(val_loader):
            
            mask = (labels != 2)
            mask4 = (bore!=2)*(mask)
            output = model(data)
            output = torch.sigmoid(output)
            mask5 = output[mask4]!=bore[mask4]
            output = (output>.5).float().cpu()
            labels = labels.cpu()
            bore = bore.cpu()
            mask = mask.cpu()
            mask4 = mask4.cpu()
            scores.append(mc(labels[mask],output[mask]))
            borescores.append(mc(bore[mask4],output[mask4]))

    return scores, borescores 



def train(model,epochs,validation_scores=0,modelname = 'model'):

    with open('/nobackup/kadonahu/smap/data/testmask.pickle','rb') as handle: testmask = pickle.load(handle)
    testmask = torch.tensor(testmask).cuda()
    
    for i in range(epochs):
        model.train()
        totloss = 0
        counter = 0
        for j,(label,data,bore) in enumerate(train_loader):
            flip = np.random.randint(2)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                
                output = model(data)
                mask2 = (bore != 2)*(label != 2)
                #loss = criterion(output[mask2],bore[mask2])+.5*local_variation_loss(output)+criterion(output[mask], label[mask]) #+ criterion(output[mask2],bore[mask2])*100# + 100*local_variation_loss(output)
                #loss =  .05*local_variation_loss(output)+criterion(output[mask], label[mask])
                #loss = .1*local_variation_loss(output)+3*criterion(output[mask2], bore.flip(flip+2)[mask2]) 
                loss = .1*local_variation_loss(output)+2*criterion(output[mask2], bore[mask2])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            totloss += loss.item()
            counter+=1
        print(i,totloss,round(totloss/counter,2))
        
        scores,borescores = validate(model)
        scores = 3*np.mean(borescores)+np.mean(scores)

        if scores >= validation_scores:
            print('New high score:',scores)
            torch.save(model.state_dict(),modelname)
            torch.save(optimizer.state_dict(), f'{modelname}opt')
            validation_scores = scores
        else:
            print('score:',scores)

    return validation_scores

def test(model,outputname):
    with torch.no_grad():
        model.eval()
        predictedlist = []
        scores = []
        borescores = []
        erascores = []
        confscores = []
        rootgrp = nc(f'/nobackup/kadonahu/smap/{outputname}.nc', "w", format="NETCDF4")
        t2 = rootgrp.createDimension("time", None)
        y = rootgrp.createDimension("y", 2000)
        x = rootgrp.createDimension("x", 2000)
        times = rootgrp.createVariable('time','i2',('time'))
        tbs = rootgrp.createVariable('tb', 'f4', ('time', 'y', 'x'),zlib=True,complevel=4,least_significant_digit = 2,fill_value=0)
        con = rootgrp.createVariable('conf', 'f4', ('time', 'y', 'x'),zlib=True,complevel=4,least_significant_digit = 2, fill_value=0)

        for k, (labels, data,bore) in enumerate(test_loader):

            mask = (labels != 2)
            mask2 = (bore != 2)
            output = model(data)
            con[k] =  torch.sigmoid(output)[0][0].cpu().float().numpy()
            
            output = torch.sigmoid(output)>.5
            output = output.float()
            scores.append(len(output[mask][(output[mask]==labels[mask])])/len(output[mask]))
            borescores.append(len(output[mask2][(output[mask2] == bore[mask2])]) / (len(bore[mask2])+1e-5))
            erascores.append(len(bore[mask2][(bore[mask2] == labels[mask2])]) / (len(bore[mask2])+1e-5))
            output[mask==False] = 2
            predicted = output[0][0].cpu().detach().numpy()
            tbs[k] = predicted
            times[k] = k
        rootgrp.close()
    return scores, borescores,erascores, confscores

def train_shuffle(model, epochs, rate, folders, validation_scores=0,modelname = 'model'):
    interval = int(epochs / rate)
    for i in range(interval):
        print(i*rate,'/',epochs)
        training.shuffle(folders)
        validation_scores = train(model, rate, validation_scores,modelname=modelname)
    return validation_scores

def local_variation_loss(data, loss_func=nn.L1Loss()):
    """Compute the local variation around each pixel.

    This loss discourages high frequency noise on borders.
    """
    # Compute vertical variation
    loss = loss_func(data[..., 1:, :], data[..., :-1, :])
    # Compute horizontal variation
    loss += loss_func(data[..., :, 1:], data[..., :, :-1])
    return loss


batchsize = 4 
print("loading")
time = 'M'
training_areas = ['eastcoast','washington','alaska'] #midus
validation_areas = ['alaska','midus']
testing_areas = ['europe']

features = [4,5,6,7,8,9,10,11,12,13]#[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
modelname = '/nobackup/kadonahu/smap/bothamsrprevdaymodel'

trainyear = [2017,2018]
testyear = [2016]
year = testyear
valyear = [2016,2018]
#training = custdata(trainyear,training_areas,features,test=False,spin=True,sub=0,time=time)
#testing = custdata(testyear,testing_areas,features,test=True,time=time)
#validation = custdata(valyear,validation_areas,features,val=True,test=True,time=time)
#train_loader =  DataLoader(dataset=training,batch_size=batchsize, shuffle=True, drop_last=True)
#test_loader =  DataLoader(dataset=testing,batch_size=1, shuffle=False, drop_last=False)
#val_loader =  DataLoader(dataset=validation,batch_size=batchsize, shuffle=False, drop_last=False)


model = UNet(len(features),1,depth=4,base_filter_bank_size=32,skip=False,bndry_dropout=True,bndry_dropout_p=.2).cuda()
model.load_state_dict(torch.load(f'continue{modelname.split("/")[-1]}'))
optimizer = optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-2)
optimizer.load_state_dict(torch.load(f'continue{modelname.split("/")[-1]}opt'))

validation_scores = 0
with open(f'{modelname}valscores.pickle','rb') as handle: validation_scores = pickle.load(handle)
print('current high score:',validation_scores)



outputname = f'{modelname.split("/")[-1]}{year[0]}{time}'
criterion = nn.BCEWithLogitsLoss()#pos_weight=torch.tensor(.25))
scaler = torch.cuda.amp.GradScaler()
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.99)

#changelr(optimizer,2)
#validation_scores = train(model,15, modelname = modelname,validation_scores=validation_scores)
#validation_scores = train_shuffle(model,15,10,training_areas,validation_scores,modelname = modelname)
#
#with open(f'{modelname}valscores.pickle','wb') as handle:  pickle.dump(validation_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
#torch.save(model.state_dict(),f'continue{modelname.split("/")[-1]}')
#torch.save(optimizer.state_dict(), f'continue{modelname.split("/")[-1]}opt')

model.load_state_dict(torch.load(modelname))
for testyear in [2016,2017,2018,2019,2020]:
   testing = custdata([testyear],testing_areas,features,test=True,time=time)

   test_loader =  DataLoader(dataset=testing,batch_size=1, shuffle=False, drop_last=False)

   outputname = f'{modelname.split("/")[-1]}{testyear}{time}'
   
   print(f'best {testyear} test')
   scores,borescores,erascores,conflist = test(model,outputname)

   print("Avg Acc:",np.mean(scores))
   print("Avg Bore:",np.mean(borescores))
   print("Avg ERA:",np.mean(erascores))


