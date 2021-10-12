import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

def create_dataset_onestep (d, time,meta_data=None):
    inputs=[]
    outputs=[]
    timestep=[]
    meta=[]
    for i in range (len(d)-1):
        #print (i)
        if time[i+1]-time[i]==20:
            data_in=d[i]
            data_out=d[i+1]
            inputs.append(data_in)
            outputs.append(data_out)
            timestep.append(time[i])
  
            if meta_data!=None:
                meta.append(meta_data[i])
    
    dataset=np.concatenate((np.asarray(inputs),np.asarray(outputs)),axis=1)
    return dataset,timestep,meta
    
def create_dataset_updatefunc (d,time):
    inputs=[]
    outputs=[]
    timestep=[]
   
 
    for i in range (0,len(d)-1):
        #print (i)
        if (time[i+1]==time[i]+20):
            data_in=d[i]
            data_out=(d[i+1,0:4]-d[i,0:4])/20
            inputs.append(data_in)
            outputs.append(data_out)
            timestep.append(time[i])
           
  
      
 
    dataset=np.concatenate((np.asarray(timestep).reshape(-1,1),np.asarray(inputs),np.asarray(outputs)),axis=1)
    return  np.asarray(timestep),dataset
    
def create_dataset_PR_nets (d,time,input_num=5):
    inputs=[]
    outputs=[]
    mom=[]
    meta=[]
    timestep=[]
   
 
    for i in range (0,len(d)-1):
        #print (i)
        if (time[i+1]==time[i]+20):
            data_in=d[i,:input_num]
            data_out=(d[i,input_num:input_num+4])
            
            mom.append(d[i,input_num+4:input_num+7])
            meta.append(d[i+1,input_num+7:])
            
            inputs.append(data_in)
            outputs.append(data_out)
            timestep.append(time[i])
           
  
      
 
    dataset=np.concatenate((np.asarray(timestep).reshape(-1,1),np.asarray(inputs),np.asarray(outputs),np.asarray(mom),np.asarray(meta)),axis=1)
    return  np.asarray(timestep),dataset

def norm_data(ds,norm_start=0,norm_end=None):
    if norm_end==None:
        norm_end=ds.shape[-1]
    means, sds = np.full(norm_end-norm_start, np.nan), np.full(norm_end-norm_start, np.nan)
    n=0
    for i in range(norm_start,norm_end):
        means[n], sds[n] = np.mean(ds[:, i]), np.std(ds[:, i])
        ds[:, i] = (ds[:, i] - means[n]) / sds[n]
        n+=1

    return means, sds, ds


def dataset_transform(dataset):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   #def norm(x):
   #    return (x-np.mean(x))/(np.std(x))
   #
   #for i in range (norm_start,dataset.shape[-1]):
   #    print (i)
   #    dataset[:,i]=norm(dataset[:,i])

    data = torch.FloatTensor(dataset).to(device)
    
    batch_size = 256
    train_split = int(len(data) * 0.8)
    test_split = len(data) - train_split
    test_dataset=data[train_split:,:]
    train_dataset=data [0:train_split, :]


    train_split=int(len(train_dataset)*0.8)
    val_split=len(train_dataset)-train_split
    train_dataset, val_dataset=random_split(train_dataset, [train_split, val_split])

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)  
    
    
    return train_dataloader,val_dataloader,test_dataloader,train_dataset,val_dataset, test_dataset