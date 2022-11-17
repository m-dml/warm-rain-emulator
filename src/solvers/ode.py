import numpy as np
import torch
from torch import nn


#test
class simulation_forecast:
    def __init__(
        self,
        arr,
        new_model,
        sim_number,
        inputs_mean,
        inputs_std,
        updates_mean,
        updates_std,
        lo_norm=None
    ):
        self.arr = arr
        self.sim_number = sim_number
        self.inputs_mean = inputs_mean
        self.inputs_std = inputs_std
        self.updates_mean = updates_mean
        self.updates_std = updates_std
        self.model = new_model
        self.moment_preds = []
        self.updates_prev = None
        self.real_updates = []
        self.lo_norm = lo_norm

    def setup(self):

        arr = self.arr.astype(np.float32)
        # self.arr=np.mean(arr[:,:,719:-1,:],axis=-1)
        arr_new = np.delete((arr[:, :, :]), [3, 6], 1)

        if len(arr_new.shape)<4:
            #print("Mask is false")
            self.test_sims = arr_new 
        else:
            k = np.ma.getmask(self.arr)
            k_new = np.delete((k[:, :, :]), [3, 6], 1)
            self.test_sims = np.ma.masked_where(k_new, arr_new)

    # For testing
    def test(self):
        self.setup()
        self.orig = np.ma.compress_rows(self.test_sims[:, 1:5, self.sim_number])
        self.sim_data = self.test_sims[0, 1:, self.sim_number]
        self.model_params = self.sim_data[-4:-1]
        self.lo = self.sim_data[0] + self.sim_data[2]
        self.model_params[0] = self.lo
        self.create_input()
        #print(self.inputs)
        #np.save("/gpfs/home/sharmas/micro-param/initial_conditions/lo_002_rm_13_nu_1.npy",self.inputs.data)
        self.model.eval()
        predictions_updates = self.model.test_step(torch.from_numpy(self.inputs))
        self.moment_calc(predictions_updates)
        #print(np.ma.compress_rows(self.test_sims[:, :, self.sim_number]).shape)
        for i in range(
            1, np.ma.compress_rows(self.test_sims[:, :, self.sim_number]).shape[0]
        ):

            self.create_input()
            self.model.eval()
            predictions_updates = self.model.test_step(torch.from_numpy(self.inputs))
            self.moment_calc(predictions_updates)

    # For Calculation of Moments
    def calc_mean(self, no_norm, means, stds):
        return (no_norm - means.reshape(-1,)) / stds.reshape(
            -1,
        )

    # For creation of inputs
    def create_input(self):
        tau = self.sim_data[2] / (self.lo)

        xc = self.sim_data[0] / (self.sim_data[1] + 1e-8)
        if self.lo_norm:
            inputs = np.concatenate(
                (
                    (self.sim_data[0:4].reshape(1, -1))/self.model_params[0],
                    tau.reshape(1, -1),
                    xc.reshape(1, -1),
                    self.model_params.reshape(1, -1),
                ),
                axis=1,
            )
        else:
            inputs = np.concatenate(
                (
                    self.sim_data[0:4].reshape(1, -1),
                    tau.reshape(1, -1),
                    xc.reshape(1, -1),
                    self.model_params.reshape(1, -1),
                ),
                axis=1,
            )
            
        # new_input_=np.concatenate((predictions_orig_[:,0:],self.model_params.reshape(1,-1),tau.reshape(1,-1),xc.reshape(1,-1)),axis=1)

        self.inputs = self.calc_mean(inputs, self.inputs_mean, self.inputs_std)
        self.inputs = np.float32(self.inputs)

    # For checking updates
    def check_updates(self):
        
        if self.updates[0, 0] > 0:
            self.updates[0, 0] = 0

        if self.updates[0, 2] < 0:
            self.updates[0, 2] = 0

        if self.updates[0, 1] > 0:
            self.updates[0, 1] = 0
        self.updates[0,0] = -self.updates[0,2]
    def check_preds(self):

        if self.preds[0, 0] < 0:
            self.preds[0, 0] = 0

        if self.preds[0, 2] < 0:
            self.preds[0, 2] = 0

        if self.preds[0, 2] > self.model_params[0]:
            self.preds[0, 2] = self.model_params[0]

        if self.preds[0, 1] < 0:
            self.preds[0, 1] = 0

        if self.preds[0, 3] < 0:
            self.preds[0, 3] = 0

        #self.preds[:, 0] = self.model_params[0] - self.preds[:, 2]

    def moment_calc(self, predictions_updates):
        self.updates = (
            predictions_updates.detach().numpy() * self.updates_std
        ) + self.updates_mean
        self.check_updates()
        if self.lo_norm:
            self.preds = (self.sim_data[0:4] + (self.updates * 20))* self.model_params[1]
        else:
            self.preds = (self.sim_data[0:4] + (self.updates * 20))
        self.check_preds()
        
        # print(self.updates)
        self.moment_preds.append(self.preds)
        self.sim_data = self.preds.reshape(
            -1,
        )
        self.updates_prev = self.updates


class SB_forecast:

    kcc = 9.44e9  # Long kernel in m3 kg-2 s-1
    kcr = 5.78  # Long kernel in m3 kg-1 s-1
    krr = 4.33

    xstar = 2.6e-10  # xstar in kg
    a_phi = 600.0
    p_phi = 0.68
    rhow = 1e3

    def __init__(self, arr, sim_num):

        # self.model_params=d[self.start_point,5:8].to('cpu').numpy()*sds[4:7].reshape(1, -1) + means[4:7].reshape(1, -1)

        self.lo = arr[0, -4, sim_num]
        self.rm = arr[0, -3, sim_num]
        self.nu = arr[0, -1, sim_num]

        self.lc = arr[0, 1, sim_num]
        self.nc = arr[0, 2, sim_num]
        self.lr = arr[0, 4, sim_num]
        self.nr = arr[0, 5, sim_num]

        self.auto = None
        self.acc = None
        self.scc = None
        self.scr = None
        self.xc0 = None

        self.predictions = []
        self.test_sims = arr
        self.sim_num = sim_num

    def SB_calc(self):
        print(np.ma.compress_rows(self.test_sims[:, :, self.sim_num]).shape)
        for i in range(
            1, np.ma.compress_rows(self.test_sims[:, :, self.sim_num]).shape[0]
        ):
            self.autoconSB()
            self.accretionSB()
            self.selfcloudSB()
            self.selfrainSB()
            self.solve_ode()
            self.predictions.append([self.lc, self.nc, self.lr, self.nr])

    @staticmethod
    def drop_mass(r):

        return 4.0 / 3.0 * np.pi * SB_forecast.rhow * (r) ** 3

    def autoconSB(self):

        nu = self.nu
        xc = self.lc / self.nc
        auto = (
            SB_forecast.kcc
            / (20 * SB_forecast.xstar)
            * (nu + 2.0)
            * (nu + 4.0)
            / (nu + 1.0) ** 2
            * self.lc**2
            * xc**2
        )
        tau = self.lr / (self.lc + self.lr + 1e-15)
        taup = np.power(tau, SB_forecast.p_phi)
        phi = SB_forecast.a_phi * taup * (1.0 - taup) ** 3
        self.auto = auto * (1.0 + phi)

    def accretionSB(self):

        tau = self.lr / (self.lc + self.lr)
        phi = (tau / (tau + 5e-4)) ** 4
        self.acc = SB_forecast.kcr * self.lc * self.lr * phi

    def selfcloudSB(self):

        self.scc = SB_forecast.kcc * (self.nu + 2.0) / (self.nu + 1.0) * self.lc**2

    def selfrainSB(self):

        self.scr = SB_forecast.krr * self.lr * self.nr

    def solve_ode(self):
        dt = 5
        xm = SB_forecast.drop_mass(self.rm)
        autoN = 1.0 / SB_forecast.xstar * self.auto
        accrN = self.acc / xm
        for i in range(4):
            self.lc = self.lc - self.auto * dt - self.acc * dt
            self.lr = self.lr + self.auto * dt + self.acc * dt
            self.nc = self.nc - accrN * dt - self.scc * dt
            self.nr = self.nr + autoN * dt - self.scr * dt
