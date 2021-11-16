import os
import sys

import numpy as np
import torch
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.plModel import LightningModel
from src.utils.FastDataloader import DataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
import os
from omegaconf import OmegaConf

GPUS = 1
N_EPOCHS = 100

if len(sys.argv)>1:
    file_name = sys.argv[1]
    
else:
    file_name = "config.yaml"

CONFIG_PATH = "conf/"
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = OmegaConf.load(file)

    return config


def cli_main():
    pl.seed_everything(42)



    data_module = DataModule(batch_size=config.batch_size,tot_len=config.tot_len,
                             test_len=config.test_len,sim_num=config.sim_num,norm=config.norm,
                             input_type=config.input_type,ic=config.ic)
    data_module.setup()

    # setting up the model:
    pl_model = LightningModel(updates_mean=data_module.updates_mean,updates_std=data_module.updates_std,
                              inputs_mean=data_module.inputs_mean,inputs_std=data_module.inputs_std,
                              batch_size=config.batch_size,beta=config.beta,
                              learning_rate=config.learning_rate,act=eval(config.act), 
                              depth=config.depth,p=config.p, norm=config.norm,
                              n_layers=config.n_layers,ns=config.ns,use_regularizer=config.use_regularizer,
                              )
    

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode='min', save_last=True)

    early_stop = EarlyStopping(monitor="val_loss", patience=10, verbose=True)

    trainer = pl.Trainer(callbacks=[checkpoint_callback, early_stop],gpus=GPUS, max_epochs=N_EPOCHS)

    trainer.fit(pl_model,data_module)

    return data_module,pl_model,trainer


if __name__ == '__main__':
    config = load_config(file_name+".yaml")
    _dm,_model, _trainer = cli_main() 
