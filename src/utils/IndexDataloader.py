import csv
import numpy as np
import os
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from src.helpers.normalizer import give_norm

class my_dataset(Dataset):
    def __init__(self,inputdata,tend,outputs,index_arr,step_size,moment_scheme):
        self.inputdata=inputdata.data
        self.tend=tend.data
        self.outputs=outputs.data
        self.index_arr=index_arr
        self.step_size=step_size
        self.moment_scheme=moment_scheme
        self.tend_new=np.empty((self.moment_scheme*2,self.step_size))
        self.out_new=np.empty((self.moment_scheme*2,self.step_size))

    def __getitem__(self,index):

        for i in range (self.step_size):
            self.tend_new[:,i] = self.tend[tuple(self.index_arr[index+i])]
            self.out_new[:,i] = self.outputs[tuple(self.index_arr[index+i])]
            
        return self.inputdata[tuple(self.index_arr[index])], self.tend_new, self.out_new
    
    def __len__(self):
        return (self.index_arr.shape[0])



class DataModule(pl.LightningDataModule):

    def __init__(self, data_dir= "/gpfs/work/sharmas/mc-snow-data/", 
                batch_size: int = 256, num_workers: int = 1, transform=None,tot_len=719,
                 sim_num=98,load_from_memory=True,moment_scheme=2,
                 step_size=1):
        super().__init__()


        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.moment_scheme = moment_scheme
        self.tot_len=tot_len
        self.sim_num=sim_num
        self.step_size=step_size
        self.load_from_memory =load_from_memory
        if self.load_from_memory==True:
            """ 
            All the array are of the shape: 
            (Time,initial_cond (tot 819),no of sim runs,[inputs/outputs/updates])
            """
     
            
            with np.load(data_dir+'/inputs_all.npz') as npz:
                self.inputs_arr=np.ma.MaskedArray(**npz)

            with np.load(data_dir+'/outputs_all.npz') as npz:
                self.outputs_arr=np.ma.MaskedArray(**npz)

            with np.load(data_dir+'/tendencies.npz') as npz:
                self.tend_arr=np.ma.MaskedArray(**npz)
        else:
            raise ValueError('Function needs to be called for calculating values from raw data!')

    def setup(self):
        self.calc_index_array()
        self.calc_norm()
        self.test_train()
        
    def calc_index_array_size(self):
        """Gives the total length of index array depending on the time length of the simulations"""
        l_in=0
        for i in range (self.tot_len):
            l=np.ma.compress_rows(self.tend_arr[:,i,0,:]).shape[0]-self.step_size+1  #put stepping here
            l_in+=l
        return l_in
        
    def calc_index_array(self):
        """Create an array of indices such that col1, col2,col3: Time,ic_sim,sim_no"""
        l_in = self.calc_index_array_size()
        self.indices_arr = np.empty((l_in*self.sim_num,3),dtype=np.int)  #(98*np.sum(time))
 
        lo=0

        sim_nums = np.arange(self.sim_num)
        for i in range (self.tot_len):
            l = np.ma.compress_rows(self.tend_arr[:,i,0,:]).shape[0]-self.step_size+1  #put stepping here
            
            time_points = np.arange(l)
            unique_sim_num = np.full(shape=l,fill_value=i,dtype=np.int)
            new_arr = np.concatenate((time_points.reshape(-1,1),unique_sim_num.reshape(-1,1)),axis=1)
            new_arr = np.vstack([new_arr]*self.sim_num)
            
            sim_num_axis = np.repeat(sim_nums,l,axis=0)
            
            indices_sim = np.concatenate((new_arr,sim_num_axis.reshape(-1,1)),axis=1)
            self.indices_arr[lo:lo+indices_sim.shape[0],:]=(indices_sim)
            lo+=indices_sim.shape[0]

    def calc_norm (self):
        self.inputs_mean = np.mean((np.mean(self.inputs_arr.data[:,:,0,:],axis=0)),axis=0)
        self.inputs_std = np.std((np.std(self.inputs_arr.data[:,:,0,:],axis=0)),axis=0)

        self.inputs_arr = (self.inputs_arr - self.inputs_mean) / self.inputs_std
       
        self.outputs_mean = np.mean((np.mean(self.outputs_arr.data[:,:,0,:],axis=0)),axis=0)
        self.outputs_std = np.std((np.std(self.outputs_arr.data[:,:,0,:],axis=0)),axis=0)
        self.outputs_arr = (self.outputs_arr - self.outputs_mean) / self.outputs_std

        self.updates_mean = np.mean((np.mean(self.tend_arr.data[:,:,0,:],axis=0)),axis=0)
        self.updates_std = np.std((np.std(self.tend_arr.data[:,:,0,:],axis=0)),axis=0)
        self.tend_arr = (self.tend_arr - self.updates_mean) / self.updates_std
        
    def test_train(self):
            
            self.dataset=my_dataset(self.inputs_arr,self.tend_arr,self.outputs_arr,self.indices_arr,self.step_size,self.moment_scheme)
           

            # Creating data indices for training and validation splits:
         
            train_size = int(0.9 * self.dataset.__len__())
            val_size = self.dataset.__len__() - train_size

            self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])

            print("Train Test Val Split Done")

    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self): #Only for quick checks
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False)

        