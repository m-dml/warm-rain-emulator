import os
import sys
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.models.plModel import LightningModel
from src.utils.IndexDataloader import DataModule

if len(sys.argv) > 1:
    file_name = sys.argv[1]

else:
    file_name = "avg_ic.yaml"

CONFIG_PATH = "conf/all_sims/averaged_sims/"


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
        lo_norm=False,
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
        pretrained_path=config.pretrained_dir,
        lo_norm=config.lo_norm,
        ro_norm=config.ro_norm,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="last_val_loss", save_top_k=1, mode="min", save_last=True
    )

    early_stop = EarlyStopping(monitor="last_val_loss", patience=50, verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stop, lr_monitor],
        devices=GPUS,
        accelerator="gpu",
        max_epochs=N_EPOCHS,
        num_sanity_val_steps=0,
    )

    trainer.fit(pl_model, data_module)

    return data_module, pl_model, trainer


if __name__ == "__main__":
    config = load_config(file_name + ".yaml")

    _dm, _model, _trainer = cli_main()
