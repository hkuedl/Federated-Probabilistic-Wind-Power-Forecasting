# Probabilistic-Wind-Power-Forecasting-An-Adaptive-Federated-Approach
>This work proposes a federated learning-based probabilistic WPF framework to utilize the data from other wind farms to construct forecasting models while preserving privacy. There are codes for five forecasting settings and data in this repository.

Codes for the paper "Probabilistic Wind Power Forecasting: An Adaptive Federated Approach".

Authors: Xiaorong Wang, Yangze Zhou


## Requirements
The must-have packages can be installed by running
```
pip install requirements.txt
```
```
conda env create -f environment.yml
```

## Experiments
There are five forecasting settings in this work and the code for these settings is organized in the same way. The results and models are saved in ```https://drive.google.com/drive/folders/17qs0H3TlKMRQcyTvJHz3r-gSU_MO5KGS?usp=drive_link```.

### Data
All the clean data for experiments are saved in ```Data/GFC12```. 

The row data can be found in ```Data/GFC12 row```.

You can also find the code for processing the data in this fold.

### Reproduction
If you want to run the proposed approach, you can run ```test.ipynb```.

If you want to reproduct the result of benchmarks, you can run ```main.ipynb```.

If you want to find the result of the table in the paper, you can refer to ```analysis.ipynb```.


