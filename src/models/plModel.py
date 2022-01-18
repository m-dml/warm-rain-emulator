import os
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.models.nnmodel import plNetwork
from src.solvers.ode import simulation_forecast,SB_forecast
from matplotlib import pyplot as plt
import seaborn as sns


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        data_module,
        updates_mean,
        updates_std,
        data_dir="/gpfs/work/sharmas/mc-snow-data/",
        batch_size=256,
        beta=0.35,
        learning_rate=2e-4,
        act=nn.ReLU(),
        loss_func=None,
        n_layers=5,
        ns=200,
        depth=9,
        p=0.25,
        inputs_mean=None,
        inputs_std=None,
        mass_cons_updates=False,
        loss_absolute=True,
        mass_cons_moments=True,
        hard_constraints_updates=True,
        hard_constraints_moments=False,
        multi_step=False,
        step_size=1,
        moment_scheme=2,
        plot_while_training=False,
        plot_all_moments=True,
        use_batch_norm=False,
        use_dropout=False
    ):
        super().__init__()
        # self.automatic_optimization = False
        self.moment_scheme = moment_scheme
        self.out_features = moment_scheme * 2
        """Necessary for keeping out_features for plModel 
        and nnmodel different when necessary"""
        out_features = self.out_features
        self.lr = learning_rate
        self.loss_func = loss_func
        self.beta = beta
        self.batch_size = batch_size

        self.loss_absolute = loss_absolute
        self.mass_cons_updates = mass_cons_updates
        self.mass_cons_moments = mass_cons_moments

        """ Using the following only makes sense for multi-step training"""
        self.hard_constraints_updates = hard_constraints_updates
        self.hard_constraints_moments = hard_constraints_moments

        if (
            self.hard_constraints_moments == True
            and self.hard_constraints_updates == False
        ):
            print(
                "Setting hard constrainsts on moments while none on updates can lead to problems"
            )

        if self.hard_constraints_moments == False and self.mass_cons_moments == True:
            print(
                "not using hard constraints on updates while choosing to conserve mass can lead to negative moments"
            )

        # For mass conservation
        if self.mass_cons_updates:
            out_features -= 1
            """The model will now only produce 3 outputs and give delLc = -delLr """

        self.multi_step = multi_step
        self.step_size = step_size
        self.num_workers = 48
        self.plot_while_training = plot_while_training
        self.plot_all_moments = plot_all_moments
        self.save_hyperparameters()
        self.data_module = data_module
        self.updates_std = torch.from_numpy(updates_std).float().to("cuda")
        self.updates_mean = torch.from_numpy(updates_mean).float().to("cuda")
        self.inputs_mean = torch.from_numpy(inputs_mean).float().to("cuda")
        self.inputs_std = torch.from_numpy(inputs_std).float().to("cuda")

        self.model = self.initialization_model(
            act, n_layers, ns, self.out_features, depth, p, use_batch_norm, use_dropout
        )

    @staticmethod
    def initialization_model(act, n_layers, ns, out_features, depth, p, use_batch_norm):
        os.chdir("/gpfs/work/sharmas/mc-snow-data/")
        model = plNetwork(act, n_layers, ns, out_features, depth, p, use_batch_norm, use_dropout)
        model.train()
        return model

    def forward(self):
        updates = self.model(self.x.float())
        self.updates = updates.float()
        self.check_values()

    def check_values(self):

        self.real_updates = (
            self.updates * self.updates_std + self.updates_mean
        )  # Un-normalize
        if self.hard_constraints_updates:

            """Checking the predicted values"""
            del_lc = torch.min(
                self.real_updates[:, 0], torch.tensor([0.0]).to(self.device)
            ).reshape(-1, 1)
            del_nc = torch.min(
                self.real_updates[:, 1], torch.tensor([0.0]).to(self.device)
            ).reshape(-1, 1)
            del_lr = torch.max(
                self.real_updates[:, 2], torch.tensor([0.0]).to(self.device)
            ).reshape(-1, 1)
            del_nr = self.real_updates[:, 3].reshape(-1, 1)
            self.real_updates = torch.cat((del_lc, del_nc, del_lr, del_nr), axis=1)
            self.updates = (
                self.real_updates - self.updates_mean
            ) / self.updates_std  # normalized and stored as updates, to be used for direct comaprison

        self.real_x = (
            self.x[:, : self.out_features] * self.inputs_std[: self.out_features]
            + self.inputs_mean[: self.out_features]
        )  # un-normalize

        self.pred_moment = self.real_x + self.real_updates * 20
        Lo = (
            self.x[:, -3] * self.inputs_std[-3] + self.inputs_mean[-3]
        )  # For total water content
        if self.hard_constraints_moments:

            """Checking moments"""
            lb = torch.zeros((1, 4)).to(self.device)
            ub_num_counts = (
                (lb.new_full((Lo.shape[0], 1), 1e10)).reshape(-1, 1).to(self.device)
            )  # Upper bounds for number counts
            ub = torch.cat(
                (Lo.reshape(-1, 1), ub_num_counts, Lo.reshape(-1, 1), ub_num_counts),
                axis=1,
            ).to(self.device)
            # creating upper bounds for all four moments

            self.pred_moment = torch.max(torch.min(self.pred_moment, ub), lb)

        if self.mass_cons_moments:

            """Best not to use if hard constraints are not used in moments"""
            self.pred_moment[:, 0] = (
                Lo - self.pred_moment[:, 2]
            )  # Lc calculated from Lr

        # self.pred_moment_norm = torch.empty(
        #     (self.pred_moment.shape), dtype=torch.float32, device=self.device
        # )
        self.pred_moment_norm = (
            self.pred_moment - self.inputs_mean[: self.out_features]
        ) / self.inputs_std[
            : self.out_features
        ]  # Normalized value of predicted moments (not updates)

    def loss_function(self, updates, y):

        if self.loss_func == "mse":
            self.criterion = torch.nn.MSELoss()
        elif self.loss_func == "mae":
            self.criterion = torch.nn.L1Loss()
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction="mean", beta=self.beta)

        # For moments
        if self.loss_absolute:
            pred_loss = self.criterion(self.pred_moment_norm, y)

        else:
            pred_loss = self.criterion(updates, self.updates)
            

        return pred_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        return optimizer

    def training_step(self, batch, batch_idx):

        """Dim self.x: batch, inputs (4 moments,self.xc,tau,3 initial conditions)
        Dim updates, y: batch, moments,step size
        y = x + updates*20"""

        self.x, updates, y = batch
        self.x = self.x.type(torch.DoubleTensor).to(self.device)
        
        self.loss_each_step, self.cumulative_loss = torch.tensor(
            (0.0), dtype=torch.float32, device=self.device
        ), torch.tensor((0.0), dtype=torch.float32, device=self.device)
        for k in range(self.step_size):

            self.forward()

            assert self.updates is not None
            self.loss_each_step = self.loss_function(
                updates[:, :, k].float(), y[:, :, k].float()
            )
            self.cumulative_loss = self.cumulative_loss + self.loss_each_step
    

            if self.step_size > 1:
                self.calc_new_x(y[:, :, k].float(), k)
            new_str = "Train_loss_" + str(k)
            self.log(new_str, self.loss_each_step)

        self.log("train_loss", self.cumulative_loss.reshape(1, 1))

        return self.cumulative_loss

    def validation_step(self, batch, batch_idx):
        print("In Val Loop")
        self.x, updates, y = batch
        self.x = self.x.type(torch.DoubleTensor).to("cuda")
        self.loss_each_step, self.cumulative_loss = torch.tensor(
            (0.0), dtype=torch.float32, device=self.device
        ), torch.tensor((0.0), dtype=torch.float32, device=self.device)
        for k in range(self.step_size):

            self.forward()
            assert self.updates is not None
            self.loss_each_step = self.loss_function(
                updates[:, :, k].float(), y[:, :, k].float()
            )
            self.cumulative_loss = self.cumulative_loss + self.loss_each_step

            if self.step_size > 1:
                self.calc_new_x(y[:, :, k].float(), k)

        self.log("val_loss", self.cumulative_loss.reshape(1, 1))
        return self.cumulative_loss

    def test_step(self, initial_moments):
        """For moment-wise evaluation as used for ODE solve"""
        with torch.no_grad():
            pred = self.model(initial_moments.float())
        return pred
    
    def on_train_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        #Give a check here for batch_idx so that it's only called once
        self.x, updates, y = batch
        self.x = self.x.type(torch.DoubleTensor).to("cuda")
        self.loss_each_step, self.cumulative_loss = torch.tensor(
            (0.0), dtype=torch.float32, device=self.device
        ), torch.tensor((0.0), dtype=torch.float32, device=self.device)
        for k in range(self.step_size):

            self.forward()

            assert self.updates is not None
            self.loss_each_step = self.loss_function(
                updates[:, :, k].float(), y[:, :, k].float()
            )
            self.cumulative_loss = self.cumulative_loss + self.loss_each_step
    

            if self.step_size > 1:
                self.calc_new_x(y[:, :, k].float(), k)
            new_str = "Train_loss_" + str(k)
            self.log(new_str, self.loss_each_step)

        self.log("train_loss", self.cumulative_loss.reshape(1, 1))
        
    def on_train_batch_end(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        
        self.x, updates, y = batch
        if self.plot_all_moments:
            loss_lc,loss_nc,loss_lr,loss_nr = self.criterion(updates[:,0], self.updates[:,0]),
            self.criterion(updates[:,1], self.updates[:,1]),self.criterion(updates[:,2], self.updates[:,2]),
            self.criterion(updates[:,3], self.updates[:,3])
            
            self.logger.experiment.add_scalars('Loss_', 
                                            {'Lc': loss_lc, 
                                            'Nc': loss_nc,
                                            'Lr':loss_lr, 
                                            'Nr': loss_nr},
                                            global_step=self.global_step)
            
        self.x = self.x.type(torch.DoubleTensor).to("cuda")
        self.pers_loss_each_step, self.pers_cumulative_loss = torch.tensor(
            (0.0), dtype=torch.float32, device=self.device
        ), torch.tensor((0.0), dtype=torch.float32, device=self.device)

        """Persistence baseline calculated here"""
        for k in range(self.step_size):
            self.updates = torch.zeros((1,4),device=self.device) 
            self.check_values()
            self.per_loss_each_step = self.loss_function(
                updates[:, :, k].float(), y[:, :, k].float()
            )
            self.pers_cumulative_loss = self.pers_cumulative_loss + self.pers_loss_each_step

            if self.step_size > 1:
                self.calc_new_x(y[:, :, k].float(), k)
            new_str = "Persistence_loss_at_step_" + str(k)
            self.log(new_str, self.pers_loss_each_step)

        self.log("Persistence_Loss", self.pers_cumulative_loss.reshape(1, 1))
        

    def training_epoch_end(self, outputs) -> None:
        """With the trained model, calculate the ODE solution"""
        nn_forecast = simulation_forecast()
        assert self.data_module.single_sim is not None
        nn_forecast.setup(self.all_data,self.model,self.data_module.single_sim, self.data_module)
        #Add the plotting function here
        
        return super().training_epoch_end(outputs)

    def calc_new_x(self, y, k):
        if self.plot_while_training:

            if (self.global_step % 2000) == 0:
                fig, figname = self.plot_preds()
                self.logger.experiment.add_figure(
                    figname + ": Step  " + str(k), fig, self.global_step
                )

        if k < self.step_size - 1:
            """A new x is calculated at every step. Takes moments as calculated
            from NN outputs (pred_moment_norm) along with other paramters
            that are fed as inputs to the network (pred_moment)"""

            self.x_old = y
            tau = self.pred_moment[:, 2] / (
                self.pred_moment[:, 2] + self.pred_moment[:, 0]
            )  # Taking tau and xc from the calculated outputs
            xc = self.pred_moment[:, 0] / (self.pred_moment[:, 1] + 1e-8)

            self.x[:, 4] = (tau - self.inputs_mean[4]) / self.inputs_std[4]
            self.x[:, 5] = (xc - self.inputs_mean[5]) / self.inputs_std[5]
            self.x[:, :4] = self.pred_moment_norm

            # Add plotting of the new x created just to see the difference
            if self.plot_while_training:
                if (self.global_step % 2000) == 0:
                    fig, figname = self.plot_new_input()
                    self.logger.experiment.add_figure(
                        figname + ": Step  " + str(k), fig, self.global_step
                    )
    def plot_moment_loss(self):
        
                
    def plot_new_input(self):
        x = (
            self.x_old * self.inputs_std[: self.out_features]
            + self.inputs_mean[: self.out_features]
        )
        x = x.cpu().detach().numpy()
        y = (
            self.x[:, : self.out_features] * self.inputs_std[: self.out_features]
            + self.inputs_mean[: self.out_features]
        )
        y = y.cpu().detach().numpy()

        var = ["Lc", "Nc", "Lr", "Nr"]
        c = ["#26235b", "#bc473a", "#812878", "#f69824"]
        sns.set_theme(style="darkgrid")
        fig = plt.figure(figsize=(15, 12))
        for i in range(4):
            ax = fig.add_subplot(2, 2, i + 1)
            plt.scatter(x[:, i], y[:, i], color=c[i])

            plt.title(var[i])
            plt.ylabel("As calculated during training")
            plt.xlabel("Original Input")
            plt.tight_layout()

            figname = "Comparison of calculated inputs vs orig "

        return fig, figname

    def plot_preds(self):
        x = self.real_x[:, : self.out_features].cpu().detach().numpy()
        y = self.pred_moment.cpu().detach().numpy()
        var = ["Lc", "Nc", "Lr", "Nr"]
        c = ["#26235b", "#bc473a", "#812878", "#f69824"]
        sns.set_theme(style="darkgrid")
        fig = plt.figure(figsize=(15, 12))
        for i in range(4):
            ax = fig.add_subplot(2, 2, i + 1)
            sns.regplot(x[:, i], y[:, i], color=c[i])

            plt.title(var[i])
            plt.ylabel("Neural Network Predictions")
            plt.xlabel("Real Value")
            plt.tight_layout()

        figname = "Mc-Snow vs ML"

        return fig, figname
