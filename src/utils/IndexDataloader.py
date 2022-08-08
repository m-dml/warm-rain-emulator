from re import S
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader


class my_dataset(Dataset):
    def __init__(self, inputdata, tend, outputs, index_arr, step_size, moment_scheme):
        self.inputdata = inputdata.data
        self.tend = tend.data
        self.outputs = outputs.data
        self.index_arr = index_arr
        self.step_size = step_size
        self.moment_scheme = moment_scheme

        """The following code differentiates between two modes of running this code:
           with masked arrays or arrays"""
        try:
            assert isinstance(self.inputdata, np.ndarray)
        except:
            self.inputdata = inputdata
            self.tend = tend
            self.outputs = outputs
        assert isinstance(self.inputdata, np.ndarray) and isinstance(self.tend, np.ndarray) and isinstance(self.outputs, np.ndarray)
            
        

    def __getitem__(self, index):
        i_time, i_ic, i_repeat = self.index_arr[index]
        
        tend_multistep = np.empty(
            (self.moment_scheme * 2, self.step_size)
        )  # tendencies
        outputs_multistep = np.empty(
            (self.moment_scheme * 2, self.step_size)
        )  # outputs (moments)
        for i_step in range(self.step_size):
            
            tend_multistep[:, i_step] = self.tend[i_time + i_step, i_ic, i_repeat]
            
            outputs_multistep[:, i_step] = self.outputs[i_time + i_step, i_ic, i_repeat]
        
        return (
            torch.from_numpy(self.inputdata[i_time, i_ic, i_repeat])
            .view(-1, 1)
            .float(),
            torch.from_numpy(tend_multistep).view(-1, self.step_size).float(),
            torch.from_numpy(outputs_multistep).view(-1, self.step_size).float(),
        )
       
    def __len__(self):
        return self.index_arr.shape[0]


def normalize_data(x, flag = None):
    """
    normalize array by all but the last dimension, return normed vals, means, sds
    """
   
    x_ = x.reshape(-1, x.shape[-1])
    m = x_.mean(axis=0)
    s = x_.std(axis=0)
    s[s == 0] = 1
    return (x - m) / s, m, s


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir="/gpfs/work/sharmas/mc-snow-data/",
        batch_size: int = 256,
        num_workers: int = 1,
        tot_len=719,
        sim_num=98,
        load_from_memory=True,
        moment_scheme=2,
        step_size=1,
        train_size=0.9,
        single_sim_num=None,
        avg_dataloader=False,
    ):
        """

        :param data_dir: directory with data
        :param batch_size:
        :param num_workers:
        :param tot_len: number of unique initial conditions
        :param sim_num: number of repeated simulations for each initial condition
        :param load_from_memory:
        :param moment_scheme:
        :param step_size: number of steps in the future to predict during training, minimum is 1
        :param train_size: fraction of data that will be used for training out of all data
        """
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.moment_scheme = moment_scheme
        self.tot_len = tot_len
        self.sim_num = sim_num
        self.step_size = step_size
        self.load_simulations = load_from_memory
        self.train_size = train_size
        self.single_sim_num = single_sim_num
        self.avg_dataloader = avg_dataloader
        self.lo_norm = True
        if self.load_simulations:
            """
            All the array are of the shape:
            (Time,initial_cond (tot 819),no of sim runs,[inputs/outputs/updates])
            if in case avg data being loaded, then shape:
            (Time, IC, 1, [inputs/outputs/updates])
            """
            try:
                with np.load(self.data_dir + "/inputs_all.npz") as npz:
                    self.inputs_arr = np.ma.MaskedArray(**npz)

                with np.load(self.data_dir + "/outputs_all.npz") as npz:
                    self.outputs_arr = np.ma.MaskedArray(**npz)

                with np.load(self.data_dir + "/tendencies.npz") as npz:
                    self.tend_arr = np.ma.MaskedArray(**npz)

                
                sim_lo = self.inputs_arr[:,:,:,0] + self.inputs_arr[:,:,:,2]
                self.inputs_arr[:,:,:,-3] = sim_lo
                print("Modified Lo!")
                

            except:
                self.inputs_arr = np.load(data_dir + "/inputs_all.npy")
                self.outputs_arr = np.load(data_dir + "/outputs_all.npy")
                self.tend_arr = np.load(data_dir + "/tendencies.npy")
                #Modify Lo
                
                sim_lo = self.inputs_arr[:,0,:,0] + self.inputs_arr[:,0,:,2]
                self.inputs_arr[:,0,:,-3] = sim_lo
                print("Modified Lo")
        else:
            raise ValueError(
                "Function needs to be called for calculating values from raw data!"
            )

            
    def setup(self):
        self.calc_norm()
        self.calc_index_array()
        self.test_train()

    def calc_index_array_size(self):
        """Gives the total length of index array depending on the time length of the simulations"""
        l_in = 0
        for i in range(self.tot_len):
            l = (
                np.ma.compress_rows(self.tend_arr[:, i, 0, :]).shape[0]
                - self.step_size
                + 1
            )  # put stepping here
            l_in += l
        return l_in

    def calc_index_array(self):
        """Create an array of indices such that col1, col2,col3: Time, ic_sim, sim_no"""
        l_in = self.calc_index_array_size()
        self.indices_arr = np.empty((l_in * self.sim_num, 3), dtype=np.int)
        lo = 0

        sim_nums = np.arange(self.sim_num)
        for i in range(self.tot_len):
            l = (
                np.ma.compress_rows(self.tend_arr[:, i, 0, :]).shape[0]
                - self.step_size
                + 1
            )

            time_points = np.arange(l)
            unique_sim_num = np.full(shape=l, fill_value=i, dtype=np.int)
            new_arr = np.concatenate(
                (time_points.reshape(-1, 1), unique_sim_num.reshape(-1, 1)), axis=1
            )
            new_arr = np.vstack([new_arr] * self.sim_num)

            sim_num_axis = np.repeat(sim_nums, l, axis=0)

            indices_sim = np.concatenate((new_arr, sim_num_axis.reshape(-1, 1)), axis=1)
            self.indices_arr[lo : lo + indices_sim.shape[0], :] = indices_sim
            lo += indices_sim.shape[0]

    def calc_norm(self):
        if self.lo_norm:
            self.inputs_arr = self.inputs_arr[:, :, :4, :]/self.inputs_arr[:, :, 4, :]
            self.outputs_arr = self.outputs_arr[:, :, :4, :]/self.inputs_arr[:, :, 4, :] #For loss calculation
        
        self.inputs_arr, self.inputs_mean, self.inputs_std = normalize_data(
            self.inputs_arr
        )
        self.outputs_arr, self.outputs_mean, self.outputs_std = normalize_data(
            self.outputs_arr
        )
        self.tend_arr, self.updates_mean, self.updates_std = normalize_data(
            self.tend_arr
        )
       
            #self.tend_arr = self.tend_arr[:, :, :, :]/self.inputs_arr[:, :, 4, :]
            
        if self.avg_dataloader:
            
            self.inputs_arr = np.expand_dims(
                np.mean(self.inputs_arr[:, :, :, :], axis=2), axis=2
            )
            self.outputs_arr = np.expand_dims(
                np.mean(self.outputs_arr[:, :, :, :], axis=2), axis=2
            )
            self.tend_arr = np.expand_dims(
                np.mean(self.tend_arr[:, :, :, :], axis=2), axis=2
            )
            self.sim_num = 1

        # if self.single_sim_num:
        #     # self.inputs_arr = np.expand_dims(
        #     #     self.inputs_arr[:, self.single_sim_num, :, :], axis=1
        #     # )
        #     # self.outputs_arr = np.expand_dims(
        #     #     self.outputs_arr[:, self.single_sim_num, :, :], axis=1
        #     # )
        #     # self.tend_arr = np.expand_dims(
        #     #     self.tend_arr[:, self.single_sim_num, :, :], axis=1
        #     # )

        #     self.tot_len = 1

    def test_train(self):

        self.dataset = my_dataset(
            self.inputs_arr,
            self.tend_arr,
            self.outputs_arr,
            self.indices_arr,
            self.step_size,
            self.moment_scheme,
        )

        # Creating data indices for training and validation splits:
        # if self.single_sim_num is not None:
        #     self.train_size = 1
        #     self.train_dataset = self.val_dataset = self.dataset

        # else:
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

    def test_dataloader(self):  # Only for quick checks
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )