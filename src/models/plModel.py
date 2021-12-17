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
                 mass_cons_loss_updates=True,loss_absolute=True,mass_cons_loss_moments=True,hard_constraints_updates=True,
                 hard_constraints_moments=True,
                 multi_step=False,step_size=1,moment_scheme=2,plot_while_training=True):
        super().__init__()
        #self.automatic_optimization = False
        self.moment_scheme= moment_scheme
        self.out_features = moment_scheme * 2
        self.lr=learning_rate
        self.loss_func = loss_func
        self.beta=beta
        self.batch_size=batch_size

        self.loss_absolute=loss_absolute
        self.mass_cons_loss_updates = mass_cons_loss_updates
        self.mass_cons_loss_moments = mass_cons_loss_moments

        #Using the following only makes sense for multi-step training
        self.hard_constraints_updates = hard_constraints_updates
        self.hard_constraints_moments = hard_constraints_moments 

        if self.hard_constraints_moments == True and self.hard_constraints_updates == False:
            print ("Setting hard constrainsts on moments while none on updates can lead to problems")

        if self.hard_constraints_moments == False and self.mass_cons_loss_moments == True:
            print("not using hard constraints on updates while choosing to conserve mass can lead to negative moments")

        
        self.multi_step=multi_step
        self.step_size=step_size
        self.num_workers=48
        self.plot_while_training=plot_while_training
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
        if self.hard_constraints_updates == True:
            
            
            real_pred = self.predictions * self.updates_std + self.updates_mean
            
            """Checking the predicted values"""
            real_pred[:,0] = torch.min(real_pred[:,0],torch.tensor([0.]).to(self.device))
            real_pred[:,1] = torch.min(real_pred[:,1] ,torch.tensor([0.]).to(self.device))
            real_pred[:,2] = torch.max(real_pred[:,2] ,torch.tensor([0.]).to(self.device))

            predictions = (real_pred - self.updates_mean) / self.updates_std
          
        return predictions
    
    def loss_function(self,pred,updates,x,y):
        if self.loss_func=="mse":
            criterion=torch.nn.MSELoss()
        elif self.loss_func=="mae":
            criterion=torch.nn.L1Loss()
        else:
            criterion=torch.nn.SmoothL1Loss(reduction='mean', beta=self.beta)
        
        
        #For mass conservation
        if self.mass_cons_loss_updates == True: 
            
            pred[:,2] = pred[:,0]
          
            

        #For moments
        if self.loss_absolute==True:
         # Removing Norm

            real_pred = pred* self.updates_std + self.updates_mean
            real_x = x[:,:self.out_features] * self.inputs_std[:self.out_features] + self.inputs_mean[:self.out_features]
            real_y = y * self.inputs_std[:self.out_features] + self.inputs_mean[:self.out_features]
            Lo = x[:,-3] *self.inputs_std[-3] +self.inputs_mean[-3] 
            
            pred_moment = real_x + real_pred*20
            self.prev_m=pred_moment
            if self.hard_constraints_moments:
                """Checking moments"""
                #pred_moment = torch.max(pred_moment,torch.tensor([0.]).to(self.device))
                pred_moment[:,0] = torch.max(pred_moment[:,0] ,torch.tensor([0.]).to(self.device))
                pred_moment[:,1] = torch.max(pred_moment[:,1] ,torch.tensor([0.]).to(self.device))
                pred_moment[:,2] = torch.max(pred_moment[:,2] ,torch.tensor([0.]).to(self.device))
                pred_moment[:,3] = torch.max(pred_moment[:,3] ,torch.tensor([0.]).to(self.device))
            self.new_m=pred_moment
            if self.mass_cons_loss_moments:
                
                #Best not to use if hard constraints are not used in moments
                pred_moment[:,0] = Lo - pred_moment[:,2]
            self.newer_m=pred_moment
            pred_moment_norm = torch.empty((y.shape), dtype=torch.float32,device = self.device)
            pred_moment_norm[:,:self.out_features] = (pred_moment - self.inputs_mean[:self.out_features]) / self.inputs_std[:self.out_features]
               

            pred_loss= criterion(pred_moment_norm,y) 

            return pred_loss, pred_moment #Returning here so it doesn't need to be calculated again in calc_new_x
        
        elif self.loss_absolute==False:
            pred_loss= criterion(updates,pred)
            
            return pred_loss 
        
        
        
        

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
                self.loss [k,:],pred_moment = self.loss_function(pred.float(), updates[:,:,k].float(),x.float(),y[:,:,k].float())
            
            
            if self.step_size > 1:
                x=(self.calc_new_x(x,pred,y[:,:,k].float(),k,pred_moment)).float()
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
                self.loss[k,:],pred_moment = self.loss_function(pred.float(), updates[:,:,k].float(),x.float(),y[:,:,k].float())
          
            if self.step_size > 1:
                x=(self.calc_new_x(x,pred,y[:,:,k].float(),k,pred_moment)).float()

        
        
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
    
    def calc_new_x(self,x,pred,y,k,pred_moment): 
        
        #un-normalize preds
        real_pred = pred* self.updates_std + self.updates_mean
        real_x = x[:,:self.out_features] * self.inputs_std[:self.out_features] + self.inputs_mean[:self.out_features]
        
        
        pred_moment = real_x + real_pred *20
        if self.hard_constraints_moments:
            """Checking moments"""
            #pred_moment = torch.max(pred_moment,torch.tensor([0.]).to(self.device))
            pred_moment[:,0] = torch.max(pred_moment[:,0] ,torch.tensor([0.]).to(self.device))
            pred_moment[:,1] = torch.max(pred_moment[:,1] ,torch.tensor([0.]).to(self.device))
            pred_moment[:,2] = torch.max(pred_moment[:,2] ,torch.tensor([0.]).to(self.device))
            pred_moment[:,3] = torch.max(pred_moment[:,3] ,torch.tensor([0.]).to(self.device))
        pred_moment_norm=torch.empty((x.shape), dtype=torch.float32, device = self.device)
        pred_moment_norm[:,:self.out_features] = (pred_moment - self.inputs_mean[:self.out_features]) / self.inputs_std[:self.out_features]
        
        tau = pred_moment[:,2]/(pred_moment[:,2]+pred_moment[:,0])  #Taking tau and xc from the calculated outputs
        xc = pred_moment[:,0]/ (pred_moment[:,1] + 1e-8)
        
        pred_moment_norm[:,4] = (tau - self.inputs_mean[4]) / self.inputs_std[4]
        pred_moment_norm[:,5] = (xc - self.inputs_mean[5]) / self.inputs_std[5]
        
        pred_moment_norm[:,6:]=x[:,6:]  #Doesn't need to be normalized/un-normalized
        
         #Add plotting here
        if self.plot_while_training:
           
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