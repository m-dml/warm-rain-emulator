import os
import sys

import numpy as np
import torch
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.plModel import LightningModel
from src.utils.IndexDataloader import DataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
import os
from omegaconf import OmegaConf






if len(sys.argv)>1:
    file_name = sys.argv[1]
    
else:
    file_name = "config"

CONFIG_PATH = "conf/"
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = OmegaConf.load(file)

    return config

GPUS = 1


def cli_main():
    pl.seed_everything(42)
    N_EPOCHS = config.max_epochs


    data_module = DataModule(batch_size=config.batch_size,tot_len=config.tot_len,
                             sim_num=config.sim_num, step_size=config.step_size,moment_scheme=config.moment_scheme)
    data_module.setup()
    # setting up the model:
    
    pl_model = LightningModel(updates_mean=data_module.updates_mean,updates_std=data_module.updates_std,
                              inputs_mean=data_module.inputs_mean,inputs_std=data_module.inputs_std,
                              batch_size=config.batch_size,beta=config.beta,
                              learning_rate=config.learning_rate,act=eval(config.act), loss_func=config.loss_func,
                              depth=config.depth,p=config.p,n_layers=config.n_layers,ns=config.ns,
                              loss_absolute=config.loss_absolute, mass_cons_loss=config.mass_cons_loss, 
                              multi_step=config.multi_step,step_size=config.step_size,moment_scheme=config.moment_scheme
                              ) 
 
    
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode='min', save_last=True)

    early_stop = EarlyStopping(monitor="val_loss", patience=10, verbose=True)

    trainer = pl.Trainer(callbacks=[checkpoint_callback, early_stop],gpus=GPUS, max_epochs=N_EPOCHS,num_sanity_val_steps=0)

    trainer.fit(pl_model,data_module)

    return data_module,pl_model,trainer


if __name__ == '__main__':
    config = load_config(file_name+".yaml")
    
    _dm,_model, _trainer = cli_main() 
