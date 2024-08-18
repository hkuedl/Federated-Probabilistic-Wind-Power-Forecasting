
import pandas as pd
from Data_loader import Dataset_Custom
import argparse
import warnings
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random 
import torch
import numpy as np

def get_data(args,flag='train'):
    if flag=='train':
        shuffle_flag=True
        drop_last=True
    elif flag == 'val':
        shuffle_flag=False
        drop_last=False
    else:
        shuffle_flag=False
        drop_last=False
    data_set=Dataset_Custom(args.root_path, data_path=args.dataset_paths, flag=flag,size=[args.seq_len,args.label_len,args.pred_len])
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)
    
    return data_set,data_loader

def plot_prob_result(result,actual,start,end):
    plt.figure(figsize=(10, 4))
    for i in range(result.shape[1]):
        print('y'+str(i))
        vars()['y'+str(i)] = result[:,i].detach().cpu().numpy()[start:end].squeeze()
    
    y=actual[start:end].squeeze()
    x = range(end-start)
    
    lists=[]
    list_names=[]
    color_list=['#c0c8de', '#7ca3c3', '#427ea3', '#2c5382']
    for i in range(result.shape[1]):
        lists.append(vars()['y'+str(i)])
        list_names.append('y'+str(i))

    # 对每个序号的元素进行排序
    sorted_lists = {}
    for name in list_names:
        sorted_lists[name + '_sort'] = []

    for i in range(len(vars()['y'+str(1)])):
        # 获取每个列表的第i个元素
        values = [lst[i] for lst in lists]
        # 对元素进行排序，并获取排序后的索引
        sorted_indices = sorted(range(len(values)), key=lambda k: values[k], reverse=True)
        # 将排序后的列表添加到sorted_lists
        for j, index in enumerate(sorted_indices):
            sorted_lists[list_names[j] + '_sort'].append(lists[index][i])


    plt.fill_between(x, sorted_lists['y0_sort'], sorted_lists['y8_sort'],  color=color_list[0], alpha=0.2)
    plt.fill_between(x, sorted_lists['y1_sort'], sorted_lists['y7_sort'], color=color_list[1], alpha=0.4)
    plt.fill_between(x, sorted_lists['y2_sort'], sorted_lists['y6_sort'],  color=color_list[2], alpha=0.6)
    plt.fill_between(x, sorted_lists['y3_sort'], sorted_lists['y5_sort'], color=color_list[3], alpha=0.8)
    plt.plot(x,sorted_lists['y4_sort'],color=color_list[3],label='Mid')
    plt.plot(x,y,color='red',label='Actual')

    plt.legend()

    # 显示图形
    plt.show()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)