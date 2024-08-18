# Probabilistic-Wind-Power-Forecasting-An-Adaptive-Federated-Approach
>[!NOTE]
>This work proposes a federated learning based probabilistic WPF framework to utilize the data from other wind farms to construct forecasting models while preserving privacy. There are codes for five forecasting setting and data in this repository.

Codes for the paper "Probabilistic Wind Power Forecasting: An Adaptive Federated Approach".

Authors: Xiaorong Wang, Yangze Zhou


## Requirements
>[!NOTE]
>Basic configuration of languages and envrionments

The must-have packages can be installed by running
```
pip install requirements.txt
```
```
conda env create -f environment.yml
```

## Experiments
>[!NOTE]
>There are five forecasting settings in this work and the code for these settings are organized in the same way. The results and models are saved in ```result```.

### Data
All the clean data for experiments are saved in ```Data/GFC12```. The row data can be find in ```Data/GFC12 row```.
You can also find the code for process the data in this fold.

### Reproduction
If you want to run the proposed approach, you can run ```test.ipynb```.
If you want to reproduct the result of benchmarks, you can run ```main.ipynb```.
If you want to find the result of the table in the paper, you can refers to ```analysis.ipynb```.


