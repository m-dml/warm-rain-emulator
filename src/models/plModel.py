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
import seaborn as sns

class LightningModel(pl.LightningModule):    

    def __init__(self, updates_mean,updates_std,data_dir= "/gpfs/work/sharmas/mc-snow-data/",batch_size=256,beta=0.35,
                 learning_rate=2e-4,act=nn.ReLU(),loss_func=None, n_layers=5,ns=200,
                 depth=9,p=0.25,inputs_mean=None,inputs_std=None,
                 mass_cons_loss_updates=False,loss_absolute=True,mass_cons_loss_moments=True,hard_constraints_updates=True,
                 hard_constraints_moments=False,
                 multi_step=False,step_size=1,moment_scheme=2,plot_while_training=False,use_batch_norm=False):
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
            
         #For mass conservation
        if self.mass_cons_loss_updates: 
            self.out_features -=1
            """Subsequent changes to the model to be added"""
          
        
        self.multi_step=multi_step
        self.step_size=step_size
        self.num_workers=48
        self.plot_while_training=plot_while_training
        self.save_hyperparameters() 
        self.updates_std=torch.from_numpy(updates_std).float().to('cuda')
        self.updates_mean=torch.from_numpy(updates_mean).float().to('cuda')
        self.inputs_mean=torch.from_numpy(inputs_mean).float().to('cuda')
        self.inputs_std=torch.from_numpy(inputs_std).float().to('cuda')
        self.predictions_pred = None
        self.predictions_actual=None

        
        self.model=self.initialization_model(act,n_layers,ns,self.out_features,depth,p,use_batch_norm)
       
        
    @staticmethod
    def initialization_model(act,n_layers,ns,out_features,depth,p,use_batch_norm):
        os.chdir ('/gpfs/work/sharmas/mc-snow-data/')
        model=plNetwork(act,n_layers,ns,out_features,depth,p,use_batch_norm)
        model.train()
        return model
        
    def forward(self):
        updates=self.model(self.x.float())
        self.updates = updates.float()
        self.check_values()
        

    def check_values(self):
        
        self.real_updates = self.updates * self.updates_std + self.updates_mean # Un-normalize
        if self.hard_constraints_updates:
            
            
            """Checking the predicted values"""
            del_lc = torch.min(self.real_updates[:,0],torch.tensor([0.]).to(self.device)).reshape(-1,1)
            del_nc = torch.min(self.real_updates[:,1] ,torch.tensor([0.]).to(self.device)).reshape(-1,1)
            del_lr = torch.max(self.real_updates[:,2] ,torch.tensor([0.]).to(self.device)).reshape(-1,1)
            del_nr = self.real_updates[:,3].reshape(-1,1)
            self.real_updates = torch.cat((del_lc,del_nc,del_lr,del_nr),axis=1)
            self.updates = (self.real_updates - self.updates_mean) / self.updates_std #normalized and stored as updates, to be used for direct comaprison

            

        self.real_x = self.x[:,:self.out_features] * self.inputs_std[:self.out_features] + self.inputs_mean[:self.out_features] #un-normalize
        
        self.pred_moment = self.real_x + self.real_updates*20
        Lo = self.x[:,-3] *self.inputs_std[-3] +self.inputs_mean[-3] #For total water
        if self.hard_constraints_moments:
           
            """Checking moments"""
            lb = torch.zeros((1,4)).to(self.device)
            ub_num_counts = (lb.new_full((Lo.shape[0], 1), 1e10)).reshape(-1,1).to(self.device)
            ub = torch.cat((Lo.reshape(-1,1),ub_num_counts,Lo.reshape(-1,1),ub_num_counts),axis = 1).to(self.device)
          
            self.pred_moment = torch.max(torch.min(self.pred_moment, ub), lb)

        
          
        if self.mass_cons_loss_moments:
            
            #Best not to use if hard constraints are not used in moments
            self.pred_moment[:,0] = Lo - self.pred_moment[:,2]
        
        self.pred_moment_norm = torch.empty((self.pred_moment.shape), dtype=torch.float32,device = self.device)
        self.pred_moment_norm[:,:self.out_features] = (self.pred_moment - self.inputs_mean[:self.out_features]) / self.inputs_std[:self.out_features]

        
        
        
    def loss_function(self,updates,y):
        
        if self.loss_func=="mse":
            criterion=torch.nn.MSELoss()
        elif self.loss_func=="mae":
            criterion=torch.nn.L1Loss()
        else:
            criterion=torch.nn.SmoothL1Loss(reduction='mean', beta=self.beta)
        
        
        #For moments
        if self.loss_absolute:
            pred_loss= criterion(self.pred_moment_norm,y) 

            return pred_loss 
        
        elif self.loss_absolute==False:
            pred_loss= criterion(updates,self.updates)
            
            return pred_loss 
        
    
    def configure_optimizers(self):
        optimizer=  torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        return optimizer


    def training_step(self, batch, batch_idx):
        
        """Dim self.x: batch, inputs (4 moments,self.xc,tau,3 initial conditions)
           Dim updates, y: batch, moments,step size"""
           
        self.x,updates,y = batch
        self.loss_each_step=[]
        self.cumulative_loss=[]
        for k in range (self.step_size):
            
            self.forward()
            assert self.updates is not None
            self.loss_each_step.append(self.loss_function(updates[:,:,k].float(),y[:,:,k].float()))
            if k > 0: 
                """First step of training"""
                self.cumulative_loss.append(self.loss_each_step[-1] + self.cumulative_loss[-1])
            else:
                self.cumulative_loss.append(self.loss_each_step[-1])
                
            if self.step_size > 1:
                self.calc_new_x(y[:,:,k].float(),k)
            new_str="Train_loss_" + str(k)
            self.log(new_str, self.loss_each_step[k])
       
        self.log("train_loss", self.cumulative_loss[-1].reshape(1,1) )
        
        return self.cumulative_loss[-1]

    
    
    def validation_step(self, batch, batch_idx):
        self.x,updates,y = batch
        self.loss_each_step=[]
        self.cumulative_loss=[]
        for k in range (self.step_size):
            
            self.forward()
            assert self.updates is not None
            self.loss_each_step.append(self.loss_function(updates[:,:,k].float(),y[:,:,k].float()))
            if k > 0:
                self.cumulative_loss.append(self.loss_each_step[-1] + self.cumulative_loss[-1])
            else:
                self.cumulative_loss.append(self.loss_each_step[-1])
            
            if self.step_size > 1:
                self.calc_new_x(y[:,:,k].float(),k)
           
       
    
        self.log("val_loss", self.cumulative_loss[-1].reshape(1,1) )
        return self.cumulative_loss[-1]




    def test_step(self, initial_moments): #For moment-wise evaluation as used for ODE solve
        with torch.no_grad():
            pred = self.model(initial_moments.float())
        return pred
    
    def calc_new_x(self,y,k): 
        if self.plot_while_training:
           
            if (self.global_step % 2000) == 0:
                fig,figname = self.plot_preds()
                self.logger.experiment.add_figure(figname + ": Step  " +str(k), fig, self.global_step)
        
        
        if k < self.step_size-1:
            self.x_old = y
            tau = self.pred_moment[:,2]/(self.pred_moment[:,2] + self.pred_moment[:,0])  #Taking tau and xc from the calculated outputs
            xc = self.pred_moment[:,0]/ (self.pred_moment[:,1] + 1e-8)
            
            self.x[:,4] = (tau - self.inputs_mean[4]) / self.inputs_std[4]
            self.x[:,5] = (xc - self.inputs_mean[5]) / self.inputs_std[5]
            self.x[:,:4] = self.pred_moment_norm

            #Add plotting of the new x created just to see the difference
            if (self.global_step % 2000) == 0:
                fig,figname = self.plot_new_input()
                self.logger.experiment.add_figure(figname + ": Step  " +str(k), fig, self.global_step)
       
        
    def plot_new_input(self):
        x = self.x_old * self.inputs_std[:self.out_features] + self.inputs_mean[:self.out_features]
        x = x.cpu().detach().numpy()
        y = self.x[:,:self.out_features] * self.inputs_std[:self.out_features] + self.inputs_mean[:self.out_features]
        y = y.cpu().detach().numpy()
        
        var=["Lc","Nc","Lr","Nr"]
        c=["#26235b","#bc473a","#812878","#f69824"]
        sns.set_theme(style="darkgrid")
        fig = plt.figure(figsize=(15, 12))
        for i in range (4):
            ax = fig.add_subplot(2,2, i + 1)
            plt.scatter(x[:,i],y[:,i],color=c[i])

            plt.title(var[i])
            plt.ylabel("As calculated during training")
            plt.xlabel("Original Input")
            plt.tight_layout()

            figname= "Comparison of calculated inputs vs orig "
        
        return fig,figname
        
    def plot_preds(self):
        x=self.real_x[:,:self.out_features].cpu().detach().numpy()
        y=self.pred_moment.cpu().detach().numpy()
        var=["Lc","Nc","Lr","Nr"]
        c=["#26235b","#bc473a","#812878","#f69824"]
        sns.set_theme(style="darkgrid")
        fig = plt.figure(figsize=(15, 12))
        for i in range (4):
            ax = fig.add_subplot(2,2, i + 1)
            sns.regplot(x[:,i],y[:,i],color=c[i])

            plt.title(var[i])
            plt.ylabel("Neural Network Predictions")
            plt.xlabel("Real Value")
            plt.tight_layout()

        figname= "Mc-Snow vs ML"
        
        return fig,figname