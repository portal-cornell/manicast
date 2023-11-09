### Environment Setup

Create a conda environment and install the dependencies by
```bash
conda create --name manicast python=3.8.16
conda activate manicast
pip install -e .
```

### Datasets

Install [AMASS](https://amass.is.tue.mpg.de/) from their official website and save it to the `./datasets` directory. Depending on the subset of relevant datasets you choose to download, you can change your desired train/val/test split by modifying the `amass_splits` variable in `data/utils/amass.py`. 

CoMaD data is already included under the `./data/comad_data` directory.

### Repo Structure
```
├── README.md
|
├── docs
│   ├── SETUP.md               <- You are here
|
├── data
│   ├── comad_data             <- CoMaD Dataset
|   	├── handover_data 
|   	├── ... 
│   ├── utils                  <- PyTorch Dataset Definitions
|   	├── comad.py 
|   	├── ... 
|
├── datasets
│   ├── amass
|     ├── ACCAD
|     ├── BioMotionLab_NTroje
|     ├── CMU
|     ├── ...
|
├── src
│   ├── pretrain.py            <- Pretrain script
│   ├── finetune.py            <- Finetune (manicast) script
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
```
