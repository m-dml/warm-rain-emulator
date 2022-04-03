import os
import sys
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.plModel import LightningModel
from src.utils.IndexDataloader import DataModule
from src.helpers.plotting import plot_simulation, calc_errors
from src.solvers.ode import simulation_forecast, SB_forecast
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from omegaconf import OmegaConf

from torch.utils.tensorboard import SummaryWriter


if len(sys.argv) > 1:
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

    data_module = DataModule(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        tot_len=config.tot_len,
        step_size=config.step_size,
        moment_scheme=config.moment_scheme,
        single_sim_num=config.single_sim_num,
        avg_dataloader=config.avg_dataloader,
        
    )
    data_module.setup()
    # setting up the model:

    pl_model = LightningModel(
        save_dir=config.save_dir,
        updates_mean=data_module.updates_mean,
        updates_std=data_module.updates_std,
        inputs_mean=data_module.inputs_mean,
        inputs_std=data_module.inputs_std,
        batch_size=config.batch_size,
        beta=config.beta,
        learning_rate=config.learning_rate,
        act=eval(config.act),
        loss_func=config.loss_func,
        depth=config.depth,
        p=config.p,
        n_layers=config.n_layers,
        ns=config.ns,
        loss_absolute=config.loss_absolute,
        mass_cons_updates=config.mass_cons_updates,
        mass_cons_moments=config.mass_cons_moments,
        hard_constraints_updates=config.hard_constraints_updates,
        hard_constraints_moments=config.hard_constraints_moments,
        multi_step=config.multi_step,
        step_size=config.step_size,
        moment_scheme=config.moment_scheme,
        use_batch_norm=config.use_batch_norm,
        use_dropout=config.use_dropout,
        single_sim_num=config.single_sim_num,
        avg_dataloader=config.avg_dataloader,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", save_top_k=1, mode="min", save_last=True
    )

    #early_stop = EarlyStopping(monitor="val_loss", patience=20, verbose=True)

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        gpus=GPUS,
        max_epochs=N_EPOCHS,
        num_sanity_val_steps=0,
        gradient_clip_val=0.5)

    trainer.fit(pl_model, data_module)

    return data_module, pl_model, trainer

def load_array():
    
    all_arr = np.load("/gpfs/work/sharmas/mc-snow-data/sim_100_data.npy")
    arr = np.mean(all_arr[:, :, :], axis=-1)
    arr = np.expand_dims(arr, axis=-1)
    all_arr = np.expand_dims(all_arr, axis=2)
    return all_arr, arr
    

if __name__ == "__main__":
    config = load_config(file_name + ".yaml")

    _dm, _model, _trainer = cli_main()

    #The code below plots ODE solutions for the trained model
    try:
        assert config.single_sim_num is not None
        all_arr, arr = load_array()

        print("Loaded Data")
        #Plotting done here
        trained_model = _model.load_from_checkpoint (config.save_dir + "/lightning_logs/version_" + os.environ["SLURM_JOB_ID"]+
                                                  "/checkpoints/last.ckpt")
        sim_num = 0
        new_forecast = simulation_forecast(
            arr,
            trained_model,
            sim_num,
            _dm.inputs_mean,
            _dm.inputs_std,
            _dm.updates_mean,
            _dm.updates_std,
        )

        new_forecast.test()

        sb_forecast = SB_forecast(arr, sim_num)
        sb_forecast.SB_calc()
        predictions_sb = np.asarray(sb_forecast.predictions).reshape(-1, 4)
        var_all = np.transpose(calc_errors(all_arr, sim_num))
        fig= plot_simulation(
            np.asarray(new_forecast.moment_preds).reshape(-1, 4),
            new_forecast.orig,
            predictions_sb,
            var_all,
            new_forecast.model_params,
            num=100,
        )
        figname = "ODE Solutions"
        dir_name = "/lightning_logs/version_"+ os.environ["SLURM_JOB_ID"]
        writer = SummaryWriter(log_dir=config.save_dir + dir_name)
        writer.add_figure(figname, fig)
        writer.close()

    except:
        pass
