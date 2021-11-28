import logging

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.models.nnmodel import plNetwork



class LightningModel(pl.LightningModule):    

    def __init__(self,updates_mean,updates_std,batch_size=256,beta=0.35,
                 learning_rate=2e-4,act=nn.ReLU(),loss_func=None, n_layers=5,ns=200,out_features=4,
                 depth=9,p=0.25, norm="Standard_Norm",inputs_mean=None,inputs_std=None,ic="All",
                 use_regularizer=False,multi_step=False,step_size=1,loss_absolute=False):
        super().__init__()
        #self.automatic_optimization = False
        self.lr=learning_rate
        self.loss_func = loss_func
        self.beta=beta
        self.batch_size=batch_size
        self.norm=norm
        self.updates_std=updates_std
        self.updates_mean=updates_mean
        self.ic=ic
        self.inputs_mean=inputs_mean
        self.inputs_std=inputs_std
        self.use_regularizer=use_regularizer
        self.model=self.initialization_model(act,n_layers,ns,out_features,depth,p)
        self.loss_absolute=loss_absolute
        self.multi_step=multi_step
        self.step_size=step_size
        self.num_workers=48
        self.save_hyperparameters() 

    @staticmethod
    def initialization_model(act,n_layers,ns,out_features,depth,p):
        
        model=plNetwork(act,n_layers,ns,out_features,depth,p)
        model.train()
        return model
        
    def forward(self, x):
        predictions=self.model(x)
        
        return predictions
       
    
    def loss_function(self,pred,updates,x,y,loss):
        if self.loss_func=="mse":
            criterion=torch.nn.MSELoss()
        elif self.loss_func=="mae":
            criterion=torch.nn.L1Loss()
        else:
            print ("This is SmoothL1")
            criterion=torch.nn.SmoothL1Loss(reduction='mean', beta=self.beta)
        
        
        if self.norm=="Rel":
            Lc_in=((x[:,0]*self.inputs_std[0])+self.inputs_mean[0])
            Lr_in=((x[:,2]*self.inputs_std[2])+self.inputs_mean[2])
            
            #Lc=pred[:,0] * Lc_in
            #Lr=pred[:,2] * Lr_in
            #Lc=pred[:,0]*x[:,2]
            #Lc=(Lc*self.updates_std[0])+self.updates_mean[0]
            #Lr=(Lr*self.updates_std[2])+self.updates_mean[2]
            loss= criterion(updates,pred)#+ criterion(Lc,(-Lr))
            print ("Loss: {}".format(loss)) 

        else:
            #For mass conservation
            Lc=(pred[:,0]*self.updates_std[0])+self.updates_mean[0]
            Lr=(pred[:,2]*self.updates_std[2])+self.updates_mean[2]
            mass_cons= criterion(Lc,(-Lr))

            #For moments
            if self.loss_absolute==True:

                real_pred=torch.empty((pred.shape), dtype=torch.float32, device = 'cuda')
                real_x=torch.empty((pred.shape), dtype=torch.float32,device = 'cuda')
                real_y=torch.empty((y.shape), dtype=torch.float32,device = 'cuda')
               
                for i in range (4): # Removing Norm
                    real_pred[:,i] = pred[:,i] * self.updates_std[i] + self.updates_mean[i]
                    real_x[:,i] = x[:,i] * self.inputs_std[i] + self.inputs_mean[i]
                    
                    

                print ("This worked")
                    
                pred_moment = real_x + real_pred*20
                pred_moment_norm = torch.empty((y.shape), dtype=torch.float32,device = 'cuda')
                
                for i in range (4): # Normalizing the moment
                    pred_moment_norm[:,i] = (pred_moment[:,i] - self.inputs_mean[i]) / self.inputs_std[i]

                #Just testing for now    
                pred_loss= criterion(pred_moment_norm,y) + criterion(updates,pred)
                 
            elif self.loss_absolute==False:
                pred_loss= criterion(updates,pred)
                
            
                        
            loss = loss + pred_loss + mass_cons 
        
        #loss= criterion(updates,preds)
        return loss
        
        
        

    def configure_optimizers(self):
        optimizer=  torch.optim.Adam(self.parameters(), lr=self.lr)
        
        return optimizer

    def training_step(self, batch, batch_idx):
        x,updates,rates,y = batch
        #loss = self.compute_loss(batch)
        
        
        loss= torch.tensor([[0]],dtype=torch.float32,device='cuda',requires_grad=True)
        
        j=0
        print ("A new batch starts here:")
        for k in range (0,self.step_size*4,4):
            print (loss)
            #self.forward(x,updates,rates,y)
            
            logits = self.forward(x.float())
            l_prev = loss
            loss = self.loss_function(logits.float(), updates[:,k:k+4].float(),x.float(),y[:,k:k+4].float(),loss)
            
            #loss.requires_grad= True
            if self.step_size > 1:
                x=(self.calc_new_x(x,logits,y[:,k:k+4].float())).float()
            new_str="Train_loss_" + str(j)
            self.log(new_str, loss-l_prev)
            j+=1
            
        # Add logging
        self.log("train_loss", loss)
        
        return loss

    
    
    def validation_step(self, batch, batch_idx):
        x,updates,rates,y = batch
        
        loss= torch.tensor([[0]],dtype=torch.float32,device='cuda',requires_grad= True)
        
        for k in range (0,int(self.step_size*4),4):
            
            #self.forward(x,updates,rates,y)
            logits = self.forward(x.float())
            loss = self.loss_function(logits.float(), updates[:,k:k+4].float(),x.float(),y[:,k:k+4].float(),loss)
            if self.step_size > 1:
                x=(self.calc_new_x(x,logits,y[:,k:k+4].float())).float()
           
        self.log("val_loss", loss)
        
        return loss




    def test_step(self, batch, batch_idx):
        x,updates,rates, y = batch
        logits = self.forward(x)
        loss = self.loss_function(logits, updates,x,y)
        
        predictions_pred.append(logits)
        predictions_actual.append(y)
        return logits,y
    
    def calc_new_x(self,x,pred,y): 
        #un-normalize logits
        real_pred=torch.empty((pred.shape), dtype=torch.float32, device = 'cuda')
        real_x=torch.empty((pred.shape), dtype=torch.float32,device = 'cuda')
        #real_y=torch.empty((pred.shape), dtype=torch.float32, device = 'cuda')
        
        for i in range (4): # Removing Norm
            real_pred[:,i] = pred[:,i] * self.updates_std[i] + self.updates_mean[i]
            real_x[:,i] = x[:,i] * self.inputs_std[i] + self.inputs_mean[i]
            #real_y [:,i] = y[:,i] * self.inputs_std[i] + self.inputs_mean[i]

        #calc new x
        
        
        pred_moment = real_x + real_pred*20
        pred_moment_norm = torch.empty((x.shape), dtype=torch.float32,device = 'cuda')
        
        #normalize x
        for i in range (4):
            pred_moment_norm[:,i] = (pred_moment[:,i] - self.inputs_mean[i]) / self.inputs_std[i]
            
        #z=torch.full((pred_moment.shape[0],), 1e-14).to('cuda')
        
        tau = pred_moment[:,2]/(pred_moment[:,2]+pred_moment[:,0])  #Taking tau and xc from the calculated outputs

        xc = pred_moment[:,0]/ (pred_moment[:,1] + 1e-10)
        
        pred_moment_norm[:,4] = (tau - self.inputs_mean[4]) / self.inputs_std[4]
        pred_moment_norm[:,5] = (xc - self.inputs_mean[5]) / self.inputs_std[5]
        
        pred_moment_norm[:,6:]=x[:,6:]  #Doesn't need to be normalized/un-normalized
        
        return pred_moment_norm