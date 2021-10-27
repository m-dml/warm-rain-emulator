import csv
import numpy as np
import os
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



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
                 batch_size: int = 256, num_workers: int = 40, transform=None,tot_len=819,
                 test_len=100,sim_num=98,norm="Standard_norm",input_type=None,ic="all",arr=None):
        super().__init__()
        
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.input_type = input_type
        self.ic=ic
        # self.inputs=np.load("/gpfs/work/sharmas/mc-snow-data/rel_update/inputs.npy")
        # self.updates=np.load("/gpfs/work/sharmas/mc-snow-data/rel_update/updates.npy")
        # self.outputs=np.load("/gpfs/work/sharmas/mc-snow-data/rel_update/outputs.npy")
        # self.meta=np.load("/gpfs/work/sharmas/mc-snow-data/rel_update/meta.npy")
        
        # self.inputs_mean=np.load("/gpfs/work/sharmas/mc-snow-data/rel_update/inputs_mean.npy")
        # self.updates_mean=np.load("/gpfs/work/sharmas/mc-snow-data/rel_update/updates_mean.npy")
        # self.outputs_mean=np.load("/gpfs/work/sharmas/mc-snow-data/rel_update/outputs_mean.npy")
        
        
        # self.inputs_std=np.load("/gpfs/work/sharmas/mc-snow-data/rel_update/inputs_std.npy")
        # self.updates_std=np.load("/gpfs/work/sharmas/mc-snow-data/rel_update/updates_std.npy")
        # self.outputs_std=np.load("/gpfs/work/sharmas/mc-snow-data/rel_update/outputs_std.npy")

        self.arr=arr
        self.test_len=test_len
        self.tot_len=tot_len
        self.sim_num=sim_num
        self.normalize=norm
    
    def setup(self):
        os.chdir(self.data_dir)
        with np.load('/gpfs/work/sharmas/mc-snow-data/big_box.npz') as npz:
            self.arr = np.ma.MaskedArray(**npz)
        self.arr=self.arr.astype(np.float32)
        self.arr=self.arr[:,:,:,:98]
        self.get_tendency()
        self.get_inputs()
        #np.save("/gpfs/work/sharmas/mc-snow-data/rel_update/inputs.npy",self.inputs)
        #np.save("/gpfs/work/sharmas/mc-snow-data/rel_update/updates.npy",self.updates)
        #np.save("/gpfs/work/sharmas/mc-snow-data/rel_update/outputs.npy",self.outputs)
        #np.save("/gpfs/work/sharmas/mc-snow-data/rel_update/meta.npy",self.meta)
        if self.input_type=="mass":
            self.inputs[:,1]=self.inputs[:,0]/self.inputs[:,1]
            self.inputs[:,3]=self.inputs[:,2]/self.inputs[:,3]

            self.outputs[:,1]=self.outputs[:,0]/self.outputs[:,1]
            self.outputs[:,3]=self.outputs[:,2]/self.outputs[:,3]
            
        self.test_train()
        

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
        print("Starting to calculate tendencies")
        if self.ic=="small":
            self.test_len=1
            #self.remove_ic()
            print("Removed problematic ICs")
        self.new_arr=np.delete((self.arr[:,:,:,:]),[3,6],1)
        self.output_tend_all=[]
        
        self.length=0
        for i in range (self.tot_len):
            #print(self.arr[0,-4,i,0])
            if self.ic=="small":
                if ((self.arr[0,-4,i,0]>0.0008) and (self.arr[0,-2,i,0]<1.5)):
                    self.length+=1
                    #print ("Going here")
                    l=(np.ma.compress_rows(self.arr[:,1:5,i,0])).shape[0]
            
                    a_arr= self.new_arr [:l-1,1:5,i,:]
                    b_arr= self.new_arr [1:l,1:5,i,:] 
                    c_arr=(b_arr-a_arr)/20
                    #print (a_arr.shape)
                    self.output_tend= c_arr.transpose(0,2,1).reshape(-1,4)
                    self.output_tend_all.append(self.output_tend)
            
                    
            
            else:
                
                l=(np.ma.compress_rows(self.arr[:,1:5,i,0])).shape[0]

                a_arr= self.new_arr [:l-1,1:5,i,:]
                b_arr= self.new_arr [1:l,1:5,i,:] 

                if self.normalize=="Rel":
                    c_arr=(b_arr-a_arr)
                    a_arr[a_arr==0]=1e-14
                    c_arr=c_arr/a_arr
                

                else:
                     c_arr=(b_arr-a_arr)/20
                self.output_tend= c_arr.transpose(0,2,1).reshape(-1,4)
                   
                

                

                self.output_tend_all.append(self.output_tend)
       
        
        
        self.output_tend_all=(np.asarray(self.output_tend_all))
        
        if self.normalize=="Rel":
            #self.updates=self.norm(np.asarray(self.output_tend_all),do_norm=0)
            #self.updates_mean=None
            #self.updates_std=None
            self.updates,self.updates_mean, self.updates_std=self.norm(np.asarray(self.output_tend_all))
        else:
            
            self.updates,self.updates_mean, self.updates_std=self.norm(np.asarray(self.output_tend_all))

        print("Calculated tendencies")
            
    
        
    def get_inputs(self):
       

        input_all=[]
        meta_all=[]
        output_all=[]
        
        print ("Starting to create inputs")
        
        for i in range (self.tot_len):
            inputs_sim=[]
            meta_sim=[]
            outputs_sim=[]

            if self.ic=="small":
                if ((self.arr[0,-4,i,0]>0.0008) and (self.arr[0,-2,i,0]<1.5)):
                    l=(np.ma.compress_rows(self.arr[:,1:5,i,0])).shape[0]
                    meta=self.new_arr[:l-1,7:13,i,:]
                    meta=meta.transpose(0,2,1).reshape(-1,6)

                    tau=self.new_arr[:l-1,3,i,:]/(self.new_arr[:l-1,3,i,:]+self.new_arr[:l-1,1,i,:])
                    xc = self.new_arr[:l-1,1,i,:]/(self.new_arr[:l-1,2,i,:])

                    outputs = self.new_arr[1:l,1:5,i,:]
                    outputs=outputs.transpose(0,2,1).reshape(-1,4)

                    sim_data=self.new_arr[:l-1,1:5,i,:]
                    sim_data_1=sim_data.transpose(0,2,1).reshape(-1,4)
                    sim_data_2=self.new_arr[:l-1,-4:-1,i,:]
                    sim_data_2=sim_data_2.transpose(0,2,1).reshape(-1,3)

                    inputs=np.concatenate((sim_data_1,tau.reshape(-1,1),xc.reshape(-1,1),sim_data_2),axis=1)
            
                 
      
                
                    input_all.append(inputs)
                    meta_all.append(meta)
                    output_all.append(outputs)
            else:
                
                l=(np.ma.compress_rows(self.arr[:,1:5,i,0])).shape[0]
                meta=self.new_arr[:l-1,7:13,i,:]
                meta=meta.transpose(0,2,1).reshape(-1,6)

                tau=self.new_arr[:l-1,3,i,:]/(self.new_arr[:l-1,3,i,:]+self.new_arr[:l-1,1,i,:])
                xc = self.new_arr[:l-1,1,i,:]/(self.new_arr[:l-1,2,i,:])

                outputs = self.new_arr[1:l,1:5,i,:]
                outputs=outputs.transpose(0,2,1).reshape(-1,4)

                sim_data=self.new_arr[:l-1,1:5,i,:]
                sim_data_1=sim_data.transpose(0,2,1).reshape(-1,4)
                sim_data_2=self.new_arr[:l-1,-4:-1,i,:]
                sim_data_2=sim_data_2.transpose(0,2,1).reshape(-1,3)

                inputs=np.concatenate((sim_data_1,tau.reshape(-1,1),xc.reshape(-1,1),sim_data_2),axis=1)

                input_all.append(inputs)
                meta_all.append(meta)
                output_all.append(outputs)
        
        self.inputs,self.inputs_mean,self.inputs_std=self.norm(np.asarray(input_all))
   
        self.meta,self.meta_mean,self.meta_std=self.norm(np.asarray(meta_all))
        print("Inputs Created")
        
        
        #self.sim_dataset_inputs=np.asarray(input_all)[:,self.tot_len-self.test_len:]
        #self.sim_dataset_meta=np.asarray(meta_all)[:,self.tot_len-self.test_len:]
    
    
    
        self.outputs,self.outputs_mean,self.outputs_std=self.norm(np.asarray(output_all))
      
    
    
        print("Created Outputs")
        
    
    def norm(self,data,do_norm=1):
        norm_data=None
        if self.length >0:
            for i in range (self.length):
            
               
                    
                k=data[i]

                if norm_data is None:
                    norm_data=k



                else:
                    norm_data=np.concatenate((norm_data,k),axis=0)
                
                
        else:
            
            for i in range (self.tot_len):
                # if i== (self.tot_len-self.test_len):

                #     print ("Testing Dataset starts here")
                #     print(norm_data.shape)  
                #     self.start_test=norm_data.shape[0]



                k=data[i]
                #new_x = np.ma.masked_array(sim_data, masked_data.mask[:,:,i,j])
                #k=np.ma.compress_rows(new_x)
                if norm_data is None:
                    norm_data=k



                else:
                    norm_data=np.concatenate((norm_data,k),axis=0)
                
                
        
        
        if do_norm==1:
            b_mean=np.mean(norm_data,axis=0)



            b_std=np.std(norm_data,axis=0)
            b_std[b_std==0]=1
            
            new_data=(norm_data-b_mean)/b_std

            return new_data,b_mean, b_std
        
        else:
            return norm_data
    
    
    def test_train(self):

         
            self.inputs=np.asarray(self.inputs) 
            self.meta=np.asarray(self.meta)
            self.outputs=np.asarray(self.outputs)
            self.updates=np.asarray(self.updates)
                     
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

                