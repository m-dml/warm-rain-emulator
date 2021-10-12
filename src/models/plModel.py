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

    def __init__(self,updates_mean,updates_std,batch_size=256,beta=0.35,learning_rate=2e-4,act=nn.ReLU(),n_layers=5,ns=200,out_features=4,depth=9,p=0.25):
        super().__init__()
        self.lr=learning_rate
        self.beta=beta
        self.batch_size=batch_size
        self.save_hyperparameters()
        
        self.updates_std=updates_std
        self.updates_mean=updates_mean
        self.model=self.initialization_model(act,n_layers,ns,out_features,depth,p)
      
    

    @staticmethod
    def initialization_model(act,n_layers,ns,out_features,depth,p):
        
        model=plNetwork(act,n_layers,ns,out_features,depth,p)
        model.train()
        return model
        
    def forward(self, x):
        predictions=self.model(x)
        
        return predictions
       
      
    def loss_function(self,pred,updates):
        criterion=torch.nn.SmoothL1Loss(reduction='mean', beta=self.beta)
       
        Lc=(pred[:,0]*self.updates_std[0])+self.updates_mean[0]
        Lr=(pred[:,2]*self.updates_std[2])+self.updates_mean[2]
        loss= criterion(updates,pred) + criterion(Lc,(-Lr))
        
        #loss= criterion(updates,preds)
        return loss
        
        
        

    def configure_optimizers(self):
        optimizer=  torch.optim.Adam(self.parameters(), lr=self.lr)
        
        return optimizer

    def training_step(self, batch, batch_idx):
        x,updates,rates,y = batch
        logits = self.forward(x)
        loss = self.loss_function(logits, updates)
        # Add logging
        self.log("train_loss", loss)
  
        return loss
    


    def validation_step(self, batch, batch_idx):
        x,updates,rates, y = batch
        logits = self.forward(x)
        loss = self.loss_function(logits, updates)
       
        self.log("val_loss", loss)
        return loss




    def test_step(self, batch, batch_idx):
        x,updates,rates, y = batch
        logits = self.forward(x)
        loss = self.loss_function(logits, updates)
        
        predictions_pred.append(logits)
        predictions_actual.append(y)
        return logits,y
    
