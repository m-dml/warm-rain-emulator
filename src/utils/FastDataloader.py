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
                batch_size: int = 256, num_workers: int = 1, transform=None,tot_len=819,
                 test_len=100,sim_num=98,norm="Standard_norm",input_type=None,ic="all",arr=None,
                 step_size=1):
        super().__init__()
        
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.input_type = input_type
        self.ic=ic
        self.arr=arr
        self.test_len=test_len
        self.tot_len=tot_len
        self.sim_num=sim_num
        self.normalize=norm
        self.step_size=step_size
        
    def setup(self):
        os.chdir(self.data_dir)
        with np.load('/gpfs/work/sharmas/mc-snow-data/big_box.npz') as npz:
            self.arr = np.ma.MaskedArray(**npz)
        self.arr=self.arr.astype(np.float32)
        self.arr=self.arr[:,:,:self.tot_len,:self.sim_num]
        self.holdout()  # For creating a holdout dataset
        self.get_tendency() #For calculating tendencies
        self.get_inputs()   #For calculating inputs, outputs
        self.test_train()   #For train/val/test split; Note: Val dataset is used as a dummy test dataset
        

    def holdout(self):
        all_id=np.random.permutation(self.arr.shape[-2])
        self.test_id=all_id[:self.test_len] #100 sims for testing later
        self.train_id=all_id[self.test_len:] 
        self.holdout_arr=self.arr[:,:,self.test_id,:]
        self.arr=self.arr[:,:,self.train_id,:]
        
   
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
                # For arranging updates in 4*step_size columns : 4 stands for 4 moments
                tot_row=l-self.step_size
                tot_col=self.step_size*4
                new_output_tend=np.empty((tot_row,tot_col,self.sim_num))
                j=0
                for k in range (0,tot_col,4):
                    new_output_tend[:,k:k+4,:]=c_arr[j:(l-self.step_size+j),:,:]
                    j+=1
                self.output_tend= new_output_tend.transpose(2,0,1).reshape(-1,tot_col)
                #self.output_tend= c_arr.transpose(0,2,1).reshape(-1,4)
                self.output_tend_all.append(self.output_tend)
       
        
        
        self.output_tend_all=(np.asarray(self.output_tend_all))
        
        if self.normalize=="Rel":
            
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
                    meta=self.new_arr[:l-self.step_size,7:13,i,:]
                    meta=meta.transpose(0,2,1).reshape(-1,6)

                    tau=self.new_arr[:l-self.step_size,3,i,:]/(self.new_arr[:l-self.add_argparse_argsstep_size,3,i,:]+self.new_arr[:l-self.step_size,1,i,:])
                    xc = self.new_arr[:l-self.step_size,1,i,:]/(self.new_arr[:l-self.step_size,2,i,:])

                    outputs = self.new_arr[1:l,1:5,i,:]
                    tot_row=l-self.step_size
                    tot_col=self.step_size*4
                    new_output=np.empty((tot_row,tot_col,self.sim_num))
                    j=1
                    for k in range (0,tot_col,4):
                        new_output[:,k:k+4,:]=outputs[j:(l-self.step_size+j),:,:]
                        j+=1
                        
                    outputs= new_output.transpose(2,0,1).reshape(-1,tot_col)
                    
                    #outputs=outputs.transpose(0,2,1).reshape(-1,4)

                    sim_data=self.new_arr[:l-self.step_size,1:5,i,:]
                    sim_data_1=sim_data.transpose(0,2,1).reshape(-1,4)
                    sim_data_2=self.new_arr[:l-self.step_size,-4:-1,i,:]
                    sim_data_2=sim_data_2.transpose(0,2,1).reshape(-1,3)

                    inputs=np.concatenate((sim_data_1,tau.reshape(-1,1),xc.reshape(-1,1),sim_data_2),axis=1)
            
                 
      
                
                    input_all.append(inputs)
                    meta_all.append(meta)
                    output_all.append(outputs)
            else:

                l=(np.ma.compress_rows(self.arr[:,1:5,i,0])).shape[0]
                meta=self.new_arr[:l-self.step_size,7:13,i,:]
                meta=meta.transpose(0,2,1).reshape(-1,6)

                tau=self.new_arr[:l-self.step_size,3,i,:]/(self.new_arr[:l-self.step_size,3,i,:]+self.new_arr[:l-self.step_size,1,i,:])
                xc = self.new_arr[:l-self.step_size,1,i,:]/(self.new_arr[:l-self.step_size,2,i,:])

                outputs = self.new_arr[1:l,1:5,i,:]
                
                # For arranging output in 4*step_size columns : 4 stands for 4 moments
                tot_row=l-self.step_size
                tot_col=self.step_size*4
                new_output=np.empty((tot_row,tot_col,self.sim_num))
                j=0
                for k in range (0,tot_col,4):
                    new_output[:,k:k+4,:]=outputs[j:(l-self.step_size+j),:,:]
                    j+=1
                    
                outputs= new_output.transpose(2,0,1).reshape(-1,tot_col)
                
                #outputs=outputs.transpose(0,2,1).reshape(-1,4)

                sim_data=self.new_arr[:l-self.step_size,1:5,i,:]
                sim_data_1=sim_data.transpose(2,0,1).reshape(-1,4)
                sim_data_2=self.new_arr[:l-self.step_size,-4:-1,i,:]
                sim_data_2=sim_data_2.transpose(2,0,1).reshape(-1,3)

                inputs=np.concatenate((sim_data_1,tau.reshape(-1,1),xc.reshape(-1,1),sim_data_2),axis=1)
        
                
    
            
                input_all.append(inputs)
                meta_all.append(meta)
                output_all.append(outputs)
                
                # l=(np.ma.compress_rows(self.arr[:,1:5,i,0])).shape[0]
                # meta=self.new_arr[:l-1,7:13,i,:]
                # meta=meta.transpose(0,2,1).reshape(-1,6)

                # tau=self.new_arr[:l-1,3,i,:]/(self.new_arr[:l-1,3,i,:]+self.new_arr[:l-1,1,i,:])
                # xc = self.new_arr[:l-1,1,i,:]/(self.new_arr[:l-1,2,i,:])

                # outputs = self.new_arr[1:l,1:5,i,:]
                
                # outputs=outputs.transpose(0,2,1).reshape(-1,4)

                # sim_data=self.new_arr[:l-1,1:5,i,:]
                # sim_data_1=sim_data.transpose(0,2,1).reshape(-1,4)
                # sim_data_2=self.new_arr[:l-1,-4:-1,i,:]
                # sim_data_2=sim_data_2.transpose(0,2,1).reshape(-1,3)

                # inputs=np.concatenate((sim_data_1,tau.reshape(-1,1),xc.reshape(-1,1),sim_data_2),axis=1)

                # input_all.append(inputs)
                # meta_all.append(meta)
                # output_all.append(outputs)
        
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
                     
            #self.start_test=int(self.inputs.shape[0]-self.inputs.shape[0]*0.1)
            #self.test_dataset=my_dataset(self.inputs[self.start_test:,:],self.meta[self.start_test:,:],self.updates[self.start_test:,:],self.outputs[self.start_test:,:])

            self.dataset=my_dataset(self.inputs[:,:],self.meta[:,:],self.updates[:,:],self.outputs[:,:])
            shuffle_dataset = True


            # Creating data indices for training and validation splits:
            

            
            train_size = int(0.9 * self.inputs.shape[0])
            #self.test_dataset=my_dataset(self.inputs)
            val_size = self.inputs.shape[0] - train_size
          

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


                