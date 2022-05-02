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
        save_dir="/gpfs/work/sharmas/mc-snow-data/new_exp",
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
        use_batch_norm=False,
        use_dropout=False,
        single_sim_num=None,
        avg_dataloader=False,
        pretrained_path=None
    ):
        super().__init__()
        self.moment_scheme = moment_scheme
        self.out_features = moment_scheme * 2
        self.save_dir = save_dir
        self.lr = learning_rate
        self.loss_func = loss_func
        self.beta = beta
        self.batch_size = batch_size

        self.loss_absolute = loss_absolute

        """ Using the following only for multi-step training"""
        self.hard_constraints_updates = hard_constraints_updates
        self.hard_constraints_moments = hard_constraints_moments
        self.mass_cons_updates = mass_cons_updates  # Serves no purpose now
        self.mass_cons_moments = mass_cons_moments

        self.multi_step = multi_step
        self.step_size = step_size
        self.num_workers = 48
        self.plot_while_training = plot_while_training
        self.plot_all_moments = plot_all_moments
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
        self.pretrained_path = pretrained_path
        self.model = self.initialization_model(
            act,
            n_layers,
            ns,
            self.out_features,
            depth,
            p,
            use_batch_norm,
            use_dropout,
            save_dir,
            pretrained_path,
        )

    @staticmethod
    def initialization_model(
        act,
        n_layers,
        ns,
        out_features,
        depth,
        p,
        use_batch_norm,
        use_dropout,
        save_dir,
        pretrained_path,
    ):
        os.chdir(save_dir)
        model = plNetwork(
            act, n_layers, ns, out_features, depth, p, use_batch_norm, use_dropout
        )
        if pretrained_path is not None:
            pretrained_dict = torch.load(pretrained_path,map_location='cpu')
            new_dict = {}

            for (k, _), (_, v) in zip(
                model.state_dict().items(), pretrained_dict["state_dict"].items()
            ):
                new_dict[k] = v
            model.load_state_dict(
                dict([(n, p) for n, p in new_dict.items()]), strict=False
            )
        model.train()
        return model

    def forward(self):
        self.updates = self.model(self.x)
        self.norm_obj = normalizer(
            self.updates,
            self.x,
            self.y,
            self.updates_mean,
            self.updates_std,
            self.inputs_mean,
            self.inputs_std,
            self.device,
            self.hard_constraints_updates,
            self.hard_constraints_moments,
        )
        (
            self.real_x,
            self.real_y,
            self.pred_moment,
            self.pred_moment_norm,
        ) = self.norm_obj.calc_preds()
        self.pred_moment, self.pred_moment_norm = self.norm_obj.set_constraints()

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
            if (self.training is True) and self.plot_all_moments is True:
                self._plot_all_moments(y, k)

        else:
            pred_loss = self.criterion(updates, self.updates)
        self.check_loss(pred_loss, k)
        return pred_loss

    def check_loss(self, pred_loss, k):
        if torch.isnan(pred_loss):
            fig, figname = self.plot_preds()
            self.logger.experiment.add_figure(
                figname + ": Step  " + str(k), fig, self.global_step
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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
            self.y = y[:, :, k].squeeze()
            self.forward()

            assert self.updates is not None

            self.loss_each_step = self.loss_function(updates[:, :, k], y[:, :, k], k)
            self.cumulative_loss = self.cumulative_loss + self.loss_each_step

            if self.step_size > 1:
                self.calc_new_x(y[:, :, k], k)
            new_str = "Train_loss_" + str(k)
            self.log(new_str, self.loss_each_step)
            self.log("train_loss", self.cumulative_loss.reshape(1, 1))

        return self.cumulative_loss

    def validation_step(self, batch, batch_idx):
        self.x, updates, y = batch

        self.x = self.x.squeeze()

        self.loss_each_step = self.cumulative_loss = torch.tensor(
            (0.0), dtype=torch.float32, device=self.device
        )
        val_preds_step = []
        for k in range(self.step_size):
            self.y = y[:, :, k].squeeze()
            self.forward()
            assert self.updates is not None
            self.loss_each_step = self.loss_function(updates[:, :, k], y[:, :, k], k)
            val_preds_step.append(self.pred_moment.cpu().numpy())
            self.cumulative_loss = self.cumulative_loss + self.loss_each_step
            if self.step_size > 1:
                self.calc_new_x(y[:, :, k], k)

        self.log("val_loss", self.cumulative_loss.reshape(1, 1))

        val_preds_step = np.asarray(val_preds_step)
        val_preds_step = np.moveaxis(val_preds_step, 0, -1)

        return {"preds": val_preds_step, "y": y.cpu().numpy()}

    def validation_epoch_end(self, validation_step_outputs):
        # validation_step_outputs is a list of dictionaries

        if self.single_sim_num is not None:
            outs_list = []
            preds_list = []
            for k, v in [
                (k, v) for x in validation_step_outputs for (k, v) in x.items()
            ]:
                if k == "preds":
                    preds_list.append(v)
                else:
                    outs_list.append(v)

            self.val_preds = np.vstack(preds_list)
            self.val_y = np.vstack(outs_list)
            """Now we have stacked outputs and predictions, no shuffle hence no rearrangement needed """
            for k in range(self.step_size):
                fig, figname = self._plot_val_outputs(
                    self.val_y[:, :, k], self.val_preds[:, :, k], k
                )

                self.logger.experiment.add_figure(
                    figname + ": Step  " + str(k + 1), fig, self.global_step
                )

    def test_step(self, initial_moments):

        """For moment-wise evaluation as used for ODE solve"""
        with torch.no_grad():
            preds = self.model(initial_moments.float())
        return preds

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

            # self.pred_moment, self.pred_moment_norm = self.norm_obj.set_constraints()
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

    def plot_preds(self, k=None):
        x = self.real_y[:, : self.out_features].cpu().detach().numpy()
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

        figname = "Mc-Snow vs ML at NaN"

        return fig, figname

    def _plot_all_moments(self, y, k):

        loss_lc = self.criterion(self.pred_moment_norm[:, 0], y[:, 0])

        loss_nc = self.criterion(self.pred_moment_norm[:, 1], y[:, 1])
        loss_lr = self.criterion(self.pred_moment_norm[:, 2], y[:, 2])
        loss_nr = self.criterion(self.pred_moment_norm[:, 3], y[:, 3])

        self.logger.experiment.add_scalars(
            "Loss_step_" + str(k + 1),
            {"Lc": loss_lc, "Nc": loss_nc, "Lr": loss_lr, "Nr": loss_nr},
            global_step=self.global_step,
        )

    def _plot_val_outputs(self, stacked_y, all_preds, k):
        stacked_y = (
            stacked_y * self.inputs_std[: self.out_features].cpu().detach().numpy()
        ) + self.inputs_mean[: self.out_features].cpu().detach().numpy()
        delta = stacked_y - all_preds
        time = np.arange(delta.shape[0])
        sns.set_theme(style="darkgrid")
        fig = plt.figure(figsize=(15, 12))
        for i in range(4):
            ax = fig.add_subplot(2, 2, i + 1)
            plt.plot(time, delta[:, i], color=self.color[i])
            plt.title(self.var[i])
            plt.ylabel("Real Moments - Predicted Moments (Norm)")
            plt.xlabel("Time")
            plt.tight_layout()
            plt.suptitle("Residuals at step " + str(k + 1))

        figname = "Residuals"

        return fig, figname
