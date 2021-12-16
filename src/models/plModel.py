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
from matplotlib import pyplot as plt


class LightningModel(pl.LightningModule):    

    def __init__(self, updates_mean,updates_std,data_dir= "/gpfs/work/sharmas/mc-snow-data/",batch_size=256,beta=0.35,
                 learning_rate=2e-4,act=nn.ReLU(),loss_func=None, n_layers=5,ns=200,
                 depth=9,p=0.25,inputs_mean=None,inputs_std=None,
                 mass_cons_loss=False,loss_absolute=False,multi_step=False,step_size=1,moment_scheme=2):
        super().__init__()
        #self.automatic_optimization = False
        self.moment_scheme= moment_scheme
        self.out_features = moment_scheme * 2
        self.lr=learning_rate
        self.loss_func = loss_func
        self.beta=beta
        self.batch_size=batch_size
        
        
        
        self.loss_absolute=loss_absolute
        self.mass_cons_loss = mass_cons_loss
        self.multi_step=multi_step
        self.step_size=step_size
        self.num_workers=48
        self.save_hyperparameters() 

        self.loss= torch.empty((self.step_size,1),dtype=torch.float32,device=self.device,requires_grad=True)
        
        self.updates_std=torch.from_numpy(updates_std).float().to('cuda')
        self.updates_mean=torch.from_numpy(updates_mean).float().to('cuda')
        self.inputs_mean=torch.from_numpy(inputs_mean).float().to('cuda')
        self.inputs_std=torch.from_numpy(inputs_std).float().to('cuda')
        self.predictions_pred = None
        self.predictions_actual=None

        
        self.model=self.initialization_model(act,n_layers,ns,self.out_features,depth,p)
       
        
    @staticmethod
    def initialization_model(act,n_layers,ns,out_features,depth,p):
        os.chdir ('/gpfs/work/sharmas/mc-snow-data/')
        model=plNetwork(act,n_layers,ns,out_features,depth,p)
        model.train()
        
        return model
        
    def forward(self, x):
        predictions=self.model(x)
        self.predictions = predictions.float()
       
        #un-normalize logits
        if self.mass_cons_loss == False:
            
            
            self.real_pred = self.predictions * self.updates_std + self.updates_mean
            
            """Checking the predicted values"""
            self.real_pred[:,0] = torch.where(self.real_pred[:,0] < 0, self.real_pred[:,0], torch.tensor([0.]).to(device='cuda'))
            self.real_pred[:,1] = torch.where(self.real_pred[:,1] < 0, self.real_pred[:,1],torch.tensor([0.]).to(device='cuda'))
            self.real_pred[:,2] = torch.where(self.real_pred[:,2] > 0, self.real_pred[:,2], torch.tensor([0.]).to(device='cuda'))

            predictions = (self.real_pred - self.updates_mean) / self.updates_std
            #predictions = give_norm(self.real_pred,torch.from_numpy(self.updates_mean).to(device='cuda'),torch.from_numpy(self.updates_std).to(device='cuda'))
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
         # Removing Norm

            real_pred = pred* self.updates_std + self.updates_mean
            real_x = x[:,:self.out_features] * self.inputs_std + self.inputs_mean
            real_y = y * self.inputs_std + self.inputs_mean
                
            pred_moment = real_x + real_pred*20
            pred_moment_norm = torch.empty((y.shape), dtype=torch.float32,device = self.device)
            pred_moment_norm[:,:self.out_features] = (pred_moment - self.inputs_mean[self.out_features]) / self.inputs_std[self.out_features]
               

            pred_loss= criterion(pred_moment_norm,y) 
                 
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
        for k in range (self.step_size):
            
            pred = self.forward(x.float())
            with torch.no_grad():
                self.loss [k,:] = self.loss_function(pred.float(), updates[:,:,k].float(),x.float(),y[:,:,k].float())
            
            
            if self.step_size > 1:
                x=(self.calc_new_x(x,pred,y[:,:,k].float(),k)).float()
            new_str="Train_loss_" + str(k)
            self.log(new_str, self.loss[k])
       
        
       
        loss_tot = self.loss.sum().reshape(1,1)
       
    
        self.log("train_loss", loss_tot)
        
        return loss_tot

    
    
    def validation_step(self, batch, batch_idx):
        x,updates,y = batch
        
        for k in range (self.step_size):
            pred = self.forward(x.float())
            with torch.no_grad():
                self.loss[k,:] = self.loss_function(pred.float(), updates[:,:,k].float(),x.float(),y[:,:,k].float())
          
            if self.step_size > 1:
                x=(self.calc_new_x(x,pred,y[:,:,k].float(),k)).float()

        
        
        loss_tot = self.loss.sum().reshape(1,1) 
        self.log("val_loss", loss_tot)
        
        return loss_tot




    def test_step(self, batch, batch_idx):
        x,updates, y = batch
        pred = self.forward(x)
        loss = self.loss_function(pred,updates,x,y)
        
        self.predictions_pred.append(pred)
        self.predictions_actual.append(y)
        return pred,y
    
    def calc_new_x(self,x,pred,y,k): 
        #un-normalize logits
           
        real_pred = pred * self.updates_std + self.updates_mean
        real_x = x[:,:self.out_features] * self.inputs_std[:self.out_features] + self.inputs_mean[:self.out_features]
           
        #calc new x
        pred_moment = real_x + real_pred*20
            
        pred_moment_norm=torch.empty((x.shape), dtype=torch.float32, device = self.device)
        pred_moment_norm[:,:self.out_features] = (pred_moment - self.inputs_mean[:self.out_features]) / self.inputs_std[:self.out_features]
        
        tau = pred_moment[:,2]/(pred_moment[:,2]+pred_moment[:,0])  #Taking tau and xc from the calculated outputs
        xc = pred_moment[:,0]/ (pred_moment[:,1] + 1e-8)
        
        pred_moment_norm[:,4] = (tau - self.inputs_mean[4]) / self.inputs_std[4]
        pred_moment_norm[:,5] = (xc - self.inputs_mean[5]) / self.inputs_std[5]
        
        pred_moment_norm[:,6:]=x[:,6:]  #Doesn't need to be normalized/un-normalized
        
         #Add plotting here
        if (self.global_step % 2000) == 0:
            fig,figname = self.plot_preds(x=real_x[:,:self.out_features],x_pred = pred_moment)
            self.logger.experiment.add_figure(figname + ": Step  " +str(k), fig, self.global_step)
        
        return pred_moment_norm
    
    def plot_preds(self,x,x_pred):
        x=x.cpu().detach().numpy()
        y=x_pred.cpu().detach().numpy()
        var=["Lc","Nc","Lr","Nr"]
        c=["#26235b","#bc473a","#812878","#f69824"]
        fig = plt.figure(figsize=(15, 12))

        for i in range (4):
            ax = fig.add_subplot(2,2, i + 1)
            plt.scatter(x[:,i],y[:,i],color=c[i])

            plt.title(var[i])
            plt.ylabel("Neural Network Predictions")
            plt.xlabel("Real Value")
            plt.tight_layout()

        figname= "Mc-Snow vs ML"
        
        return fig,figname