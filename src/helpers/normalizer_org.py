import numpy as np
import torch


class normalizer:
    """Will take the output of the NN as input,input and means and stds"""

    def __init__(
        self,
        updates,
        inputs,
        outputs,
        updates_mean,
        updates_std,
        inputs_mean,
        inputs_std,
        device,
        lo_norm=False,
        
        hard_constraints_updates=True,
        hard_constraints_moments=True,
        mass_cons_moments=True,
        out_features=4,
        
    ):

        self.updates = updates
        self.x = inputs
        self.y = outputs
        self.updates_mean = updates_mean
        self.updates_std = updates_std
        self.inputs_mean = inputs_mean
        self.inputs_std = inputs_std
        self.out_features = out_features
        self.real_x = self.real_y = (
            self.real_updates
        ) = self.pred_moment = self.pred_moment_norm = None
        self.hard_constraints_updates = hard_constraints_updates
        self.hard_constraints_moments = hard_constraints_moments
        self.mass_cons_moments = mass_cons_moments
        self.device = device
        self.lo_norm = lo_norm
    def calc_preds(self):
        self.real_updates = (
            self.updates * self.updates_std + self.updates_mean
        )  # Un-normalize

        self.real_x = (
            self.x[:, : self.out_features] * self.inputs_std[: self.out_features]
            + self.inputs_mean[: self.out_features]
        )
        lo = (self.x[:, -3] * self.inputs_std[-3]) + self.inputs_mean[-3]
        ro = (self.x[:, -2] * self.inputs_std[-2]) + self.inputs_mean[-2]
        if self.lo_norm:
            #removing Lo norm from real_x values
          
            self.real_x = self.real_x[:, :4] * lo.reshape(-1, 1)
            
        
            #Scaling with respect to the corresponding Lo
        self.pred_moment = (self.real_x + self.real_updates * 20)
        # if self.mass_cons_moments:
        #     print(lo.shape)
        #     print(self.pred_moment.shape)
        #     """Best not to use if hard constraints are not used in moments"""
        #     self.pred_moment[:, 0] = (
        #         lo - self.pred_moment[:, 2]
        #     )  # Lc calculated from Lr
            
        if self.lo_norm:   
            self.pred_moment_norm =  self.pred_moment/(lo).reshape(-1,1)
            self.pred_moment_norm = ((
        self.pred_moment_norm - self.inputs_mean[: self.out_features]) / self.inputs_std[: self.out_features])
            
        else:
            self.pred_moment_norm = ((
            self.pred_moment - self.inputs_mean[: self.out_features]) / self.inputs_std[: self.out_features])
            
        self.real_y = (
            self.y[:, : self.out_features] * self.inputs_std[: self.out_features]
            + self.inputs_mean[: self.out_features]
        )
        if self.lo_norm:
            self.real_y[:, : self.out_features] =  self.real_y[:, : self.out_features] * lo.reshape(-1,1)
       
        return self.real_x, self.real_y, self.pred_moment, self.pred_moment_norm, lo, ro

    def set_constraints(self):
        assert self.updates is not None
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
        # self.real_x = (
        #     self.x[:, : self.out_features] * self.inputs_std[: self.out_features]
        #     + self.inputs_mean[: self.out_features]
        # )  # un-normalize
        # self.pred_moment = self.real_x + self.real_updates * 20
        
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

            self.pred_moment = torch.max(torch.min(self.pred_moment, ub), lb)

        if self.mass_cons_moments:

            """Best not to use if hard constraints are not used in moments"""
            self.pred_moment[:, 0] = (
                Lo - self.pred_moment[:, 2]
            )  # Lc calculated from Lr
            
        if self.lo_norm:   
            self.pred_moment_norm[:, 0] = self.pred_moment/Lo.reshape(-1,1)
            self.pred_moment_norm[:, 0] =  (self.pred_moment_norm[:, 0] - self.inputs_mean[2: 3].reshape(-1,1)) / self.inputs_std[2: 3].reshape(-1,1)
            #self.pred_moment_norm = (self.pred_moment_norm - self.inputs_mean[: self.out_features]) / self.inputs_std[: self.out_features]
           
        else:
            self.pred_moment_norm = (
                self.pred_moment - self.inputs_mean[: self.out_features]
            ) / self.inputs_std[
                : self.out_features
            ]  # Normalized value of predicted moments (not updates)

        return self.pred_moment, self.pred_moment_norm