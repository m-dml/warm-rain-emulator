Code for training SuperdropNet in PyTorch Lightning. 

## Training Data
Data can be found at: https://zenodo.org/records/10054101

## Installation and training
Create a new Conda-environment. We provide an envrironment.yaml file for dependencies.
```
  conda env create -f environment.yaml
```
For training, adjust the parameters in ``confs/example_config.yaml`` according to your system and run ``train_save.py``.
For submitting a batch job, adjuts the paremeters in in the shell script ``train_strand.sh`` or create your own shell script and submit a job using 
```
sbatch train_strand.sh
```
### Multi-step training

Adjust the paramter ``step_size`` in the config file. 
To provide a warm start to the network weights, point the parameter ``pretrained_dir`` to the converged model path at a previous ``step_size``.

