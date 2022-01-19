import numpy as np
import pytorch_lightning as pl
import torch
import pickle
from torch.utils.data import Dataset, DataLoader


class my_dataset(Dataset):
    def __init__(self, data, index_arr, listing, step_size, moment_scheme):
        self.inputdata = data[listing["inputs"]]
        self.updates = data[listing["updates"]]
        self.outputs = data[listing["outputs"]]

        self.index_arr = index_arr
        self.step_size = step_size
        self.moment_scheme = moment_scheme

    def __getitem__(self, index):
        i_time, i_ic, i_repeat = self.index_arr[index]
        updates_multistep = np.empty(
            (self.moment_scheme * 2, self.step_size)
        )  # tendencies
        outputs_multistep = np.empty(
            (self.moment_scheme * 2, self.step_size)
        )  # outputs (moments)
        for i_step in range(self.step_size):
            updates_multistep[:, i_step] = self.updates[i_ic][i_repeat][i_time + i_step]
            outputs_multistep[:, i_step] = self.outputs[i_ic][i_repeat][i_time + i_step]
      
        return (
            torch.from_numpy(self.inputdata[i_ic][i_repeat][i_time]).view(-1,1).float(),
            torch.from_numpy(updates_multistep).view(-1,self.step_size).float(),
            torch.from_numpy(outputs_multistep).view(-1,self.step_size).float()
        )

    def __len__(self):
        return self.index_arr.shape[0]


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir="/gpfs/work/sharmas/mc-snow-data/",
        batch_size: int = 256,
        num_workers: int = 1,
        tot_len=719,
        sim_num=98,
        moment_scheme=2,
        step_size=1,
        train_size=0.9,
    ):
        super().__init__()

        self.data_dir = "/gpfs/work/sharmas/mc-snow-data/"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.moment_scheme = moment_scheme
        self.tot_len = tot_len
        self.sim_num = sim_num
        self.step_size = step_size

        self.train_size = train_size
        self.listing = {"inputs": 0, "outputs": 1, "updates": 2}
        with open(data_dir + "all_sim_data.pkl", "rb") as f:
            self.all_data = pickle.load(f)
        """Loads the dataset containing a nested ragged list. Dim as follows: 
        self.all_data[quantity][initial_conditions][no of times same ICs repeated][Time]. 
        The time dimension varies among sims with different ICs.
        For the first dim [quantity]: see self.listing"""
    def setup(self):
        self.inputs_mean, self.inputs_std = self.calc_means_stds(var="inputs")
        self.calc_norm(self.inputs_mean, self.inputs_std, var="inputs")
        self.updates_mean, self.updates_std = self.calc_means_stds(var="updates")
        self.calc_norm(self.updates_mean, self.updates_std, var="updates")
        self.calc_norm(
            m=self.inputs_mean[
                : self.moment_scheme * 2,
            ],
            s=self.inputs_std[
                : self.moment_scheme * 2,
            ],
            var="outputs",
        )
        self.calc_index_array()
        self.test_train()
        
    def calc_means_stds(self, var):  # for dealing with ragged lists
        """Works by first calculating sum and mean, followed by deviation from mean"""
        l = self.all_data[self.listing[var]][0][0].shape[-1]
        tot_sum = np.zeros((l))
        tot_num = 0
        for ic in range(self.tot_len):
            for rep in range(self.sim_num):
                sim = self.all_data[self.listing[var]][ic][rep]

                tot_sum += np.sum(sim, axis=0)
                tot_num += sim.shape[0]

        m = tot_sum / tot_num

        tot_sqr_sum = np.zeros((l))
        for ic in range(self.tot_len):
            for rep in range(self.sim_num):
                sim = self.all_data[self.listing[var]][ic][rep]
                tot_sqr_sum += np.sum(((sim - m) ** 2), axis=0)

        s = np.sqrt(tot_sqr_sum / tot_num)
        return m, s

    def calc_norm(self, m, s, var):
        sim = self.all_data[self.listing[var]][0][0]
        assert m.all() and s.all() is not None
        for ic in range(self.tot_len):
            for rep in range(self.sim_num):
                sim = self.all_data[self.listing[var]][ic][rep]

                self.all_data[self.listing[var]][ic][rep] = (sim - m) / s

    

    def calc_index_array_size(self):
        """Gives the total length of index array depending on the time length of the simulations"""
        l_in = 0
        for i in range(self.tot_len):
            l = (
                self.all_data[0][i][0][:].shape[0] - self.step_size + 1
            )  # put stepping here
            l_in += l
        return l_in

    def calc_index_array(self):
        """Create an array of indices such that col1, col2,col3: Time, ic_sim, sim_no"""
        l_in = self.calc_index_array_size()
        self.index_arr = np.empty((l_in * self.sim_num, 3), dtype=np.int)

        lo = 0

        sim_nums = np.arange(self.sim_num)
        for i in range(self.tot_len):
            l = self.all_data[0][i][0][:].shape[0] - self.step_size + 1

            time_points = np.arange(l)
            unique_sim_num = np.full(shape=l, fill_value=i, dtype=np.int)
            new_arr = np.concatenate(
                (time_points.reshape(-1, 1), unique_sim_num.reshape(-1, 1)), axis=1
            )
            new_arr = np.vstack([new_arr] * self.sim_num)

            sim_num_axis = np.repeat(sim_nums, l, axis=0)

            indices_sim = np.concatenate((new_arr, sim_num_axis.reshape(-1, 1)), axis=1)
            self.index_arr[lo : lo + indices_sim.shape[0], :] = indices_sim
            lo += indices_sim.shape[0]

    def test_train(self):

        self.dataset = my_dataset(
            self.all_data,
            self.index_arr,
            self.listing,
            self.step_size,
            self.moment_scheme,
        )

        # Creating data indices for training and validation splits:

        train_size = int(self.train_size * self.dataset.__len__())
        val_size = self.dataset.__len__() - train_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )

        print("Train Test Val Split Done")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
