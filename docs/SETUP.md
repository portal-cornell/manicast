### Environment Setup

Create a conda environment and install the dependencies by
```bash
conda create --name manicast python=3.8.16
conda activate manicast
pip install -r requirements.txt
export PYTHONPATH=$(pwd) // path to parent directory
```


### Repo Structure
```
├── README.md
|
├── docs
│   ├── SETUP.md               <- You are here
|
├── data
│   ├── comad_data             <- CoMaD Dataset
|   	├── chopping_mixing_data 
|   	├── ... 
│   ├── utils                  <- PyTorch Dataset Definitions
|   	├── comad.py 
|   	├── ... 
|
├── datasets
│   ├── amass                  <- AMASS Dataset
|   	├── ACCAD 
|   	├── ...
|
├── src
│   ├── pretrain.py            <- Pretrain script
│   ├── finetune.py            <- Finetune (manicast) script
│   ├── cost_aware_finetune.py <- Finetune (cost-weighted regression) script
│
├── model
│   ├── manicast.py            <- Model definition
│
├── model_checkpoints          <- Pretrained checkpoints
│   ├── ...
|
├── eval
│   ├── handover.py            <- Object Handover evaluation script
│   ├── reactive_stirring.py   <- Reactive Stirring evaluation script
│   ├── test_comad.py          <- CoMaD Forecasting evaluation script
│
|
env
```