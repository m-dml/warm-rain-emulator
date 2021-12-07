import logging
import os
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.models.nnmodel import plNetwork
from src.helpers.normalizer import give_norm, remove_norm


class LightningModel(pl.LightningModule):    

    def __init__(self,updates_mean,updates_std,batch_size=256,beta=0.35,
                 learning_rate=2e-4,act=nn.ReLU(),loss_func=None, n_layers=5,ns=200,out_features=4,
                 depth=9,p=0.25, norm="Standard_Norm",inputs_mean=None,inputs_std=None,
                 use_regularizer=False,multi_step=False,step_size=1,loss_absolute=False,mass_cons_loss=False):
        super().__init__()
        #self.automatic_optimization = False
        self.lr=learning_rate
        self.loss_func = loss_func
        self.beta=beta
        self.batch_size=batch_size
        self.norm=norm
       
        self.use_regularizer=use_regularizer
        self.model=self.initialization_model(act,n_layers,ns,out_features,depth,p)
        self.loss_absolute=loss_absolute
        self.mass_cons_loss = mass_cons_loss
        self.multi_step=multi_step
        self.step_size=step_size
        self.num_workers=48
        self.save_hyperparameters() 
        
        self.updates_std=updates_std
        self.updates_mean=updates_mean
        self.inputs_mean=inputs_mean
        self.inputs_std=inputs_std
        self.predictions_pred = None
        self.predictions_actual=None
        
    @staticmethod
    def initialization_model(act,n_layers,ns,out_features,depth,p):
        
        model=plNetwork(act,n_layers,ns,out_features,depth,p)
        model.train()
        return model
        
    def forward(self, x):
        predictions=self.model(x)
        predictions = predictions.float()
        #un-normalize logits
        if self.mass_cons_loss == False:
            self.real_pred=torch.empty((predictions.shape), dtype=torch.float32, device = 'cuda')
            
            for i in range (4): # Removing Norm
                self.real_pred[:,i] = remove_norm(self.predictions[:,i],self.updates_mean[:,i],self.updates_std[:,i])
            
            """Checking the predicted values"""
            self.real_pred[:,0] = torch.where(self.real_pred[:,0] < 0, self.real_pred[:,0], torch.tensor([0.]).to(device='cuda'))
            self.real_pred[:,1] = torch.where(self.real_pred[:,1] < 0, self.real_pred[:,1],torch.tensor([0.]).to(device='cuda'))
            self.real_pred[:,2] = torch.where(self.real_pred[:,2] > 0, self.real_pred[:,2], torch.tensor([0.]).to(device='cuda'))

            #predictions = (self.real_pred - torch.from_numpy(self.updates_mean).to(device='cuda')) / torch.from_numpy(self.updates_std).to(device='cuda')
            predictions = give_norm(arr=self.real_pred,means=self.updates_mean.to(device='cuda'),stds=self.updates_std.to(device='cuda'))
        return predictions
    
    def loss_function(self,pred,updates,x,y):
        if self.loss_func=="mse":
            criterion=torch.nn.MSELoss()
        elif self.loss_func=="mae":
            criterion=torch.nn.L1Loss()
        else:
           
            criterion=torch.nn.SmoothL1Loss(reduction='mean', beta=self.beta)
        
        
        #For mass conservation
        if self.mass_cons_loss == True: 
            
            Lc=(pred[:,0]*self.updates_std[0])+self.updates_mean[0]
            
            Lr=(pred[:,2]*self.updates_std[2])+self.updates_mean[2]
            mass_cons= criterion(Lc,(-Lr))
          
            
        else:
            
            mass_cons = 0
            

        #For moments
        if self.loss_absolute==True:

            real_pred=torch.empty((pred.shape), dtype=torch.float32, device = 'cuda')
            real_x=torch.empty((pred.shape), dtype=torch.float32,device = 'cuda')
            real_y=torch.empty((y.shape), dtype=torch.float32,device = 'cuda')
            
            for i in range (4): # Removing Norm
                real_pred[:,i] = remove_norm(pred[:,i],self.updates_mean[:,i],self.updates_std[:,i])
                #real_pred[:,i] = pred[:,i] * self.updates_std[i] + self.updates_mean[i]
                real_x[:,i] = remove_norm(x[:,i],self.inputs_mean[:,i],self.inputs_std[:,i])
                #real_x[:,i] = x[:,i] * self.inputs_std[i] + self.inputs_mean[i]
                real_y[:,i] = remove_norm(y[:,i],self.inputs_mean[i],self.inputs_std[i])
                
            pred_moment = real_x + real_pred*20
            #pred_moment_norm = torch.empty((y.shape), dtype=torch.float32,device = 'cuda')

              
            pred_loss= criterion(pred_moment,real_y) 
                 
        elif self.loss_absolute==False:
            pred_loss= criterion(updates,pred)
                
            
                        
        loss_net= pred_loss + mass_cons 
        
        
        return loss_net
        
        

    def configure_optimizers(self):
        optimizer=  torch.optim.Adam(self.parameters(), lr=self.lr)
        
        return optimizer


    def training_step(self, batch, batch_idx):
        """Dim x: batch, inputs (4 moments,xc,tau,3 initial conditions)
           Dim updates, y: batch, moments,step size"""
           
        x,updates,y = batch
        loss= torch.empty((self.step_size,1),dtype=torch.float32,device='cuda',requires_grad=True)
        
    
        
        for k in range (self.step_size):
            
            pred = self.forward(x.float())
           
            with torch.no_grad():
                loss [k,:] = self.loss_function(pred.float(), updates[:,:,k].float(),x.float(),y[:,:,k].float())
            
            
            if self.step_size > 1:
                x=(self.calc_new_x(x,pred,y[:,:,k].float())).float()
            new_str="Train_loss_" + str(k)
            self.log(new_str, loss[k])
           
        with torch.enable_grad():
            loss_tot = loss.sum().reshape(1,1)
       
    
        self.log("train_loss", loss_tot)
        
        return loss_tot

    
    
    def validation_step(self, batch, batch_idx):
        x,updates,y = batch
        loss= torch.empty((self.step_size,1),dtype=torch.float32,device='cuda',requires_grad=True)
       
        for k in range (self.step_size):
            pred = self.forward(x.float())
            with torch.no_grad():
                loss[k,:] = self.loss_function(pred.float(), updates[:,:,k].float(),x.float(),y[:,:,k].float())
          
            if self.step_size > 1:
                x=(self.calc_new_x(x,pred,y[:,:,k].float())).float()

        
        
        loss_tot = loss.sum().reshape(1,-1) 
        self.log("val_loss", loss_tot)
        
        return loss_tot




    def test_step(self, batch, batch_idx):
        x,updates,rates, y = batch
        pred = self.forward(x)
        loss = self.loss_function(pred, updates,x,y)
        
        self.predictions_pred.append(pred)
        self.predictions_actual.append(y)
        return pred,y
    
    def calc_new_x(self,x,pred,y): 
        #un-normalize logits
        real_pred=torch.empty((pred.shape), dtype=torch.float32, device = 'cuda')
        real_x=torch.empty((pred.shape), dtype=torch.float32,device = 'cuda')
       
        
        for i in range (4): # Removing Norm
            real_pred[:,i] = remove_norm(pred[:,i],self.updates_mean[:,i],self.updates_std[:,i])
            real_x[:,i] = remove_norm(x[:,i],self.inputs_mean[:,i],self.inputs_std[:,i])
            #real_pred[:,i] = pred[:,i] * self.updates_std[i] + self.updates_mean[i]
            #real_x[:,i] = x[:,i] * self.inputs_std[i] + self.inputs_mean[i]
           

        #calc new x
        pred_moment = real_x + real_pred*20
        pred_moment_norm = torch.empty((x.shape), dtype=torch.float32,device = 'cuda')
        
        #normalize x
        for i in range (4):
            pred_moment_norm[:,i] = (pred_moment[:,i] - self.inputs_mean[i]) / self.inputs_std[i]
            
       
        
        tau = pred_moment[:,2]/(pred_moment[:,2]+pred_moment[:,0])  #Taking tau and xc from the calculated outputs
        xc = pred_moment[:,0]/ (pred_moment[:,1] + 1e-14)
        
        pred_moment_norm[:,4] = (tau - self.inputs_mean[4]) / self.inputs_std[4]
        pred_moment_norm[:,5] = (xc - self.inputs_mean[5]) / self.inputs_std[5]
        
        pred_moment_norm[:,6:]=x[:,6:]  #Doesn't need to be normalized/un-normalized
        
        return pred_moment_norm