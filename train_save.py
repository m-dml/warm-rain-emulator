import os


import numpy as np
import torch
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.plModel import LightningModel
from src.utils.plDataloader import DataModule
import pytorch_lightning as pl

from pytorch_lightning import Trainer



GPUS = 1
N_EPOCHS = 100

def cli_main():
    pl.seed_everything(42)



    data_module = DataModule(batch_size=256,num_workers=40)
    data_module.setup()

    # setting up the model:
    pl_model = LightningModel(updates_mean=data_module.updates_mean,updates_std=data_module.updates_std)

    if GPUS < 1:
        callbacks = None
    else:
        callbacks = [pl.callbacks.GPUStatsMonitor()]


    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=2, mode='min', save_last=True)

    trainer = pl.Trainer(callbacks=callbacks, checkpoint_callback=checkpoint_callback, gpus=GPUS, max_epochs=N_EPOCHS,distributed_backend="dp")

    trainer.fit(pl_model,data_module)

    return data_module,pl_model,trainer


if __name__ == '__main__':
     _dm,_model, _trainer = cli_main() 
