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

if len(sys.argv)>1:
    n_layers = int(sys.argv[1])
    ns=int(sys.argv[2])
else:
    n_layers=5
    ns=200

GPUS = 1
N_EPOCHS = 100

def cli_main():
    pl.seed_everything(42)



    data_module = DataModule(batch_size=256,num_workers=1,tot_len=819,ic="small")
    data_module.setup()

    # setting up the model:
    pl_model = LightningModel(updates_mean=data_module.updates_mean,updates_std=data_module.updates_std,
                              inputs_mean=data_module.inputs_mean,inputs_std=data_module.inputs_std, norm="Rel",
                              n_layers=n_layers,ns=ns,ic="small")

    if GPUS < 1:
        callbacks = None
    else:
        callbacks = [pl.callbacks.GPUStatsMonitor()]


    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode='min', save_last=True)

    early_stop = EarlyStopping(monitor="val_loss", patience=10, verbose=True)

    trainer = pl.Trainer(callbacks=[checkpoint_callback, early_stop],gpus=GPUS, max_epochs=N_EPOCHS)

    trainer.fit(pl_model,data_module)

    return data_module,pl_model,trainer


if __name__ == '__main__':
     _dm,_model, _trainer = cli_main() 

