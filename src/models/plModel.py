import os
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.models.nnmodel import plNetwork
from matplotlib import pyplot as plt
import seaborn as sns
from src.helpers.normalizer import normalizer

class LightningModel(pl.LightningModule):
    def __init__(
        self,
        updates_mean,
        updates_std,
        inputs_mean,
        inputs_std,
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
        mass_cons_updates=True,
        loss_absolute=True,
        mass_cons_moments=True,
        hard_constraints_updates=True,
        hard_constraints_moments=False,
        multi_step=False,
        step_size=1,
        moment_scheme=2,
        plot_while_training=False,
        plot_all_moments=True,
        calc_persistence_loss=True,
        use_batch_norm=False,
        use_dropout=False,
        single_sim_num=None,
        avg_dataloader=False,
    ):
        super().__init__()
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
        self.calc_persistence_loss = calc_persistence_loss
        self.single_sim_num = single_sim_num
        self.avg_dataloader = avg_dataloader
        self.save_hyperparameters()

        self.updates_std = torch.from_numpy(updates_std).float().to("cuda")
        self.updates_mean = torch.from_numpy(updates_mean).float().to("cuda")
        self.inputs_mean = torch.from_numpy(inputs_mean).float().to("cuda")
        self.inputs_std = torch.from_numpy(inputs_std).float().to("cuda")

        # Some plotting stuff
        self.color = ["#26235b", "#bc473a", "#812878", "#f69824"]
        self.var = ["Lc", "Nc", "Lr", "Nr"]

        self.model = self.initialization_model(
            act, n_layers, ns, self.out_features, depth, p, use_batch_norm, use_dropout
        )

    @staticmethod
    def initialization_model(
        act, n_layers, ns, out_features, depth, p, use_batch_norm, use_dropout
    ):
        os.chdir("/gpfs/work/sharmas/mc-snow-data/")
        model = plNetwork(
            act, n_layers, ns, out_features, depth, p, use_batch_norm, use_dropout
        )
        model.train()
        return model

    def forward(self):
        updates = self.model(self.x)
        self.updates = updates.float()
        self.norm_obj= normalizer(self.updates,
                                    self.x,
                                    self.updates_mean, self.updates_std, 
                                    self.inputs_mean, self.inputs_std)
        self.real_x, self.pred_moment, self.pred_moment_norm = self.norm_obj.calc_preds()
       
    def loss_function(self, updates, y, k=None):

        if self.loss_func == "mse":
            self.criterion = torch.nn.MSELoss()
        elif self.loss_func == "mae":
            self.criterion = torch.nn.L1Loss()
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction="mean", beta=self.beta)

        # For moments
        if self.loss_absolute:

            pred_loss = self.criterion(self.pred_moment_norm, y)
            try:
                assert (self.training is True) and (self.global_step % 50 == 0)
                assert k is not None and self.plot_all_moments is True
                self._plot_all_moments(y, k)
            except:
                pass

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
        self.x = self.x.squeeze()

        self.loss_each_step = self.cumulative_loss = torch.tensor(
            (0.0), dtype=torch.float32, device=self.device
        )
        for k in range(self.step_size):

            self.forward()

            assert self.updates is not None

            self.loss_each_step = self.loss_function(updates[:, :, k], y[:, :, k], k)
            self.cumulative_loss = self.cumulative_loss + self.loss_each_step

            if self.step_size > 1:
                self.calc_new_x(y[:, :, k], k)
            new_str = "Train_loss_" + str(k)
            self.log(new_str, self.loss_each_step)

        try:

            assert self.calc_persistence_loss is True
            self._persistence_log(batch)
            self.logger.experiment.add_scalars(
                "Persistence vs Training Loss",
                {
                    "Persistence": self.pers_cumulative_loss,
                    "Training": self.cumulative_loss,
                },
                global_step=self.global_step,
            )
        except:

            self.log("train_loss", self.cumulative_loss.reshape(1, 1))

        return self.cumulative_loss

    def validation_step(self, batch, batch_idx):
        self.x, updates, y = batch
        self.x = self.x.squeeze()

        self.loss_each_step = self.cumulative_loss = torch.tensor(
            (0.0), dtype=torch.float32, device=self.device
        )
        for k in range(self.step_size):

            self.forward()
            assert self.updates is not None
            self.loss_each_step = self.loss_function(updates[:, :, k], y[:, :, k], k)
            self.cumulative_loss = self.cumulative_loss + self.loss_each_step
            if self.step_size > 1:
                self.calc_new_x(y[:, :, k], k)

        self.log("val_loss", self.cumulative_loss.reshape(1, 1))
        return self.cumulative_loss

    def test_step(self, initial_moments):
        """For moment-wise evaluation as used for ODE solve"""
        with torch.no_grad():
            preds = self.model(initial_moments.float())
        return preds

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx) -> None:
        # Give a check here for batch_idx so that it's only called once

        try:
            assert self.global_step == 0

            self.x, updates, y = batch

            self.x = self.x.squeeze().to(self.device)
            self.loss_each_step = self.cumulative_loss = torch.tensor(
                (0.0), dtype=torch.float32, device=self.device
            )
            for k in range(self.step_size):

                self.forward()

                assert self.updates is not None
                self.loss_each_step = self.loss_function(
                    updates[:, :, k].to(self.device), y[:, :, k].to(self.device)
                )
                self.cumulative_loss = self.cumulative_loss + self.loss_each_step

                if self.step_size > 1:
                    self.calc_new_x(y[:, :, k], k)

            # print("Here")
            self.log("train_loss", self.cumulative_loss.reshape(1, 1))

        except:
            pass

    def _plot_all_moments(self, y, k):

        loss_lc = self.criterion(self.pred_moment_norm[:, 0], y[:, 0])

        loss_nc = self.criterion(self.pred_moment_norm[:, 1], y[:, 1])
        loss_lr = self.criterion(self.pred_moment_norm[:, 2], y[:, 2])
        loss_nr = self.criterion(self.pred_moment_norm[:, 3], y[:, 3])

        self.logger.experiment.add_scalars(
            "Loss_step_" + str(k),
            {"Lc": loss_lc, "Nc": loss_nc, "Lr": loss_lr, "Nr": loss_nr},
            global_step=self.global_step,
        )

    def _persistence_log(self, batch):

        self.x, updates, y = batch

        self.x = self.x.squeeze().to(self.device)
        self.pers_loss_each_step = self.pers_cumulative_loss = torch.tensor(
            (0.0), dtype=torch.float32, device=self.device
        )

        """Persistence baseline calculated here"""
        for k in range(self.step_size):
            self.updates = torch.zeros((1, 4), device=self.device)
            self.check_values()

            self.per_loss_each_step = self.loss_function(
                updates[:, :, k].to(self.device), y[:, :, k].to(self.device)
            )

            self.pers_cumulative_loss = (
                self.pers_cumulative_loss + self.pers_loss_each_step
            )

            if self.step_size > 1:
                self.calc_new_x(y[:, :, k], k)

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
            self.pred_moment, self.pred_moment_norm = self.norm_obj.set_constraints()
            new_x = torch.empty_like(self.x)
            tau = self.pred_moment[:, 2] / (
                self.pred_moment[:, 2] + self.pred_moment[:, 0]
            )  # Taking tau and xc from the calculated outputs
            xc = self.pred_moment[:, 0] / (self.pred_moment[:, 1] + 1e-8)

            new_x[:, 4] = (tau - self.inputs_mean[4]) / self.inputs_std[4]
            new_x[:, 5] = (xc - self.inputs_mean[5]) / self.inputs_std[5]
            new_x[:, :4] = self.pred_moment_norm
            new_x[:, 6:] = self.x[:, 6:]
            self.x = new_x

    def plot_preds(self):
        x = self.real_x[:, : self.out_features].cpu().detach().numpy()
        y = self.pred_moment.cpu().detach().numpy()
        sns.set_theme(style="darkgrid")
        fig = plt.figure(figsize=(15, 12))
        for i in range(4):
            ax = fig.add_subplot(2, 2, i + 1)
            sns.regplot(x[:, i], y[:, i], color=self.color[i])
            plt.title(self.var[i])
            plt.ylabel("Neural Network Predictions")
            plt.xlabel("Real Value")
            plt.tight_layout()

        figname = "Mc-Snow vs ML"

        return fig, figname
