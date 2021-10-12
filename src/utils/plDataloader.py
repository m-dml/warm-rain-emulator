import csv
import numpy as np
import os
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.utils import AvgplDataloader


class my_dataset(Dataset):
    def __init__(self,inputdata,metadata,updatedata,outputdata):
        self.inputdata=inputdata
        self.updatedata=updatedata
        self.metadata=metadata
        self.outputdata=outputdata
               
    def __getitem__(self, index):
        return self.inputdata[index],self.updatedata[index],self.metadata[index], self.outputdata[index]
    def __len__(self):
        return (self.inputdata.shape[0])

class DataModule(pl.LightningDataModule):

    def __init__(self, data_dir= "/gpfs/work/sharmas/mc-snow-data/", 
                 batch_size: int = 256, num_workers: int = 40, transform=None,tot_len=819,test_len=100,sim_num=98):
        super().__init__()
        
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

        #self.inputs=np.load("/gpfs/work/sharmas/mc-snow-data/inputs.npy")
        #self.updates=np.load("/gpfs/work/sharmas/mc-snow-data/updates.npy")
        #self.outputs=np.load("/gpfs/work/sharmas/mc-snow-data/outputs.npy")
        #self.meta=np.load("/gpfs/work/sharmas/mc-snow-data/meta.npy")
        
        self.arr=None
        self.test_len=test_len
        self.tot_len=tot_len
        self.sim_num=sim_num
    
    def setup(self):
        os.chdir(self.data_dir)
        with np.load('/gpfs/work/sharmas/mc-snow-data/big_box.npz') as npz:
            self.arr = np.ma.MaskedArray(**npz)
        self.arr=self.arr.astype(np.float32)
        self.get_tendency()
        self.get_inputs()
        #np.save("/gpfs/work/sharmas/mc-snow-data/rel_update/inputs.npy",self.inputs)
        #np.save("/gpfs/work/sharmas/mc-snow-data/rel_update/updates.npy",self.updates)
        #np.save("/gpfs/work/sharmas/mc-snow-data/rel_update/outputs.npy",self.outputs)
        #np.save("/gpfs/work/sharmas/mc-snow-data/rel_update/meta.npy",self.meta)
        
        self.test_train()
        #data_module = AvgplDataloader.DataModule(batch_size=256,num_workers=40)

        #data_module.setup()
        #self.updates_mean=data_module.updates_mean
        #self.updates_std=data_module.updates_std
        #self.outputs_std=data_module.outputs_std
        #self.outputs_mean=data_module.outputs_mean
        #self.inputs_mean=data_module.inputs_mean
        #self.inputs_std=data_module.inputs_std
        

        np.save("/gpfs/work/sharmas/mc-snow-data/rel_update/inputs_mean.npy",self.inputs_mean)
        np.save("/gpfs/work/sharmas/mc-snow-data/rel_update/updates_mean.npy",self.updates_mean)
        np.save("/gpfs/work/sharmas/mc-snow-data/rel_update/outputs_mean.npy",self.outputs_mean)
        
        
        np.save("/gpfs/work/sharmas/mc-snow-data/rel_update/inputs_std.npy",self.inputs_std)
        np.save("/gpfs/work/sharmas/mc-snow-data/rel_update/updates_std.npy",self.updates_std)
        np.save("/gpfs/work/sharmas/mc-snow-data/rel_update/outputs_std.npy",self.outputs_std)


        
        
        
    def calc_mask(self,unmaksed,start=0):
        print(unmaksed.shape)
        cols=unmaksed.shape[3]
        rows=unmaksed.shape[2]
        m=(self.arr[start:rows+start,:cols,:,:])
        print(m.shape)
        print(unmaksed.shape)
        return m
        
    #Works well
    def get_tendency(self):
        #arr_new=np.delete(self.arr, [3,6], 1)
        
        output_tend_all=[]
        for i in range (self.sim_num):
            output_tend=[]
            for j in range(819):
                
                   
                #sim_data=self.arr[:,:,j,i]
                sim_data=np.delete(np.ma.compress_rows(self.arr[:,:,j,i]),[3,6],1)
                output_sim=[]
                for t in range (1,len(sim_data)):
                    output_sim.append((sim_data[t,1:5]-sim_data[t-1,1:5])/(20*(sim_data[t-1,1:5])))

                output_tend.append(output_sim)

            output_tend_all.append(output_tend)
       
        self.st=np.asarray(output_tend_all)
        #mask_array=self.calc_mask(np.asarray(output_tend_all))
     
        self.sim_dataset_updates=np.asarray(output_tend_all)[:,self.tot_len-self.test_len]
        
        
        self.updates,self.updates_mean, self.updates_std=self.norm(np.asarray(output_tend_all))

        print("Calculated tendencies")
            
        
    def get_inputs(self):
        #arr_new=np.delete(self.arr, [3,6], 1)

        input_all=[]
        meta_all=[]
        output_all=[]
        

        for i in range (self.sim_num):
            inputs_sim=[]
            meta_sim=[]
            outputs_sim=[]
            for j in range(819):
                
                sim_data=np.delete(np.ma.compress_rows(self.arr[:,:,j,i]),[3,6],1)
                #sim_data=arr_new[:,:,j,i]
                #sim_data=np.ma.compress_rows(sim_data)
                inputs=[]

                meta=sim_data[:-1,7:13]

                tau=sim_data[:-1,3]/(sim_data[:-1,3]+sim_data[:-1,1])
                xc=sim_data[:-1,1]/sim_data[:-1,2]

                inputs=np.concatenate((sim_data[:-1,1:5],tau.reshape(-1,1),xc.reshape(-1,1),sim_data[:-1,-4:-1]),axis=1)
                outputs=sim_data[1:,1:5]
                inputs_sim.append(inputs)
                meta_sim.append(meta)
                outputs_sim.append(outputs)
                
            input_all.append(inputs_sim)
            meta_all.append(meta_sim)
            output_all.append(outputs_sim)
        #call norm fucntion here and return to self values
        
        #mask_array=self.calc_mask(np.asarray(input_all))
        self.inputs,self.inputs_mean,self.inputs_std=self.norm(np.asarray(input_all))
        #self.inputs=self.norm(np.asarray(input_all),mask_array,do_norm=0)
        
        #mask_array=self.calc_mask(np.asarray(meta_all))
        self.meta,self.meta_mean,self.meta_std=self.norm(np.asarray(meta_all))
        print("Inputs Created")
        
        
        self.sim_dataset_inputs=np.asarray(input_all)[:,self.tot_len-self.test_len:]
        self.sim_dataset_meta=np.asarray(meta_all)[:,self.tot_len-self.test_len:]
    
    
        #mask_array=self.calc_mask(np.asarray(output_all),start=1)
        self.outputs,self.outputs_mean,self.outputs_std=self.norm(np.asarray(output_all))
        self.sim_dataset_output=np.asarray(output_all)[:,self.tot_len-self.test_len:]
    
    
        print("Created Outputs")
        
    
    def norm(self,data,do_norm=1):
        norm_data=None
        for i in range (819):
            if i== (self.tot_len-self.test_len):
       
                print ("Testing Dataset starts here")
                print(norm_data.shape)  
                self.start_test=norm_data.shape[0]
            for j in range (self.sim_num):
               
                    
                k=data[j,i]
                #new_x = np.ma.masked_array(sim_data, masked_data.mask[:,:,i,j])
                #k=np.ma.compress_rows(new_x)
                if norm_data is None:
                    norm_data=k
                    
                    
                    
                else:
                    norm_data=np.concatenate((norm_data,k),axis=0)
                    
                   
        
        
        if do_norm==1:
            b_mean=np.mean(norm_data,axis=0)



            b_std=np.std(norm_data,axis=0)

            new_data=(norm_data-b_mean)/b_std

            return new_data,b_mean, b_std
        
        else:
            return norm_data
    
    
    def test_train(self):
         
            
            self.start_test=int(self.inputs.shape[0]-self.inputs.shape[0]*0.1)
            
            self.test_dataset=my_dataset(self.inputs[self.start_test:,:],self.meta[self.start_test:,:],self.updates[self.start_test:,:],self.outputs[self.start_test:,:])
            self.dataset=my_dataset(self.inputs[:self.start_test,:],self.meta[:self.start_test,:],self.updates[:self.start_test,:],self.outputs[:self.start_test,:])
            
        
      
            
            shuffle_dataset = True


            # Creating data indices for training and validation splits:
            

            
            train_size = int(0.9 * self.start_test)
            val_size = self.start_test - train_size
          

            self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])
        
            print("Train Test Val Split Done")
    
    
  
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False)

                
        
        
