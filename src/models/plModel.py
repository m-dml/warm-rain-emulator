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
                 learning_rate=2e-4,act=nn.ReLU(),n_layers=5,ns=200,out_features=4,
                 depth=9,p=0.25,norm="Standard_Norm",inputs_mean=None,inputs_std=None,ic="All",
                 use_regularizer=False,multi_step=False,step_size=None,loss_absolute=False):
        super().__init__()
        self.lr=learning_rate
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
        self.save_hyperparameters()      
    

    @staticmethod
    def initialization_model(act,n_layers,ns,out_features,depth,p):
        
        model=plNetwork(act,n_layers,ns,out_features,depth,p)
        model.train()
        return model
        
    def forward(self, x):
        predictions=self.model(x)
        
        return predictions
       
      
    def loss_function(self,pred,updates,x,y):
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
            Lc=(pred[:,0]*self.updates_std[0])+self.updates_mean[0]
            Lr=(pred[:,2]*self.updates_std[2])+self.updates_mean[2]
            mass_cons= criterion(Lc,(-Lr))
            if self.loss_absolute==True:
                real_pred=(pred[:,:]*self.updates_std.reshape(1,-1)) + self.updates_mean.reshape(1,-1)
                real_x=(x[:,:]*self.inputs_std.reshape(1,-1)) + self.inputs_mean.reshape(1,-1)
                pred_moment=real_x + real_pred*20
                pred_loss= criterion(pred_moment,y)
                 
            elif self.loss_absolute==False:
                pred_loss= criterion(updates,pred)
                
            L1_reg = torch.tensor(0., requires_grad=True)
            if self.use_regularizer==True:
                for name, param in self.model.named_parameters():
                    if 'weight' in name:
                        L1_reg = L1_reg + torch.norm(param, 1)
                        
            loss= pred_loss + mass_cons + 10e-4 * L1_reg
        
        #loss= criterion(updates,preds)
        return loss
        
        
        

    def configure_optimizers(self):
        optimizer=  torch.optim.Adam(self.parameters(), lr=self.lr)
        
        return optimizer

    def training_step(self, batch, batch_idx):
        x,updates,rates,y = batch
        logits = self.forward(x)
        loss = self.loss_function(logits, updates,x,y)
        # Add logging
        self.log("train_loss", loss)
  
        return loss
    


    def validation_step(self, batch, batch_idx):
        x,updates,rates, y = batch
        logits = self.forward(x)
        loss = self.loss_function(logits, updates,x,y)
       
        self.log("val_loss", loss)
        return loss




    def test_step(self, batch, batch_idx):
        x,updates,rates, y = batch
        logits = self.forward(x)
        loss = self.loss_function(logits, updates,x,y)
        
        predictions_pred.append(logits)
        predictions_actual.append(y)
        return logits,y
    
