import os
import pandas as pd
from Data_loader import Dataset_Custom
import argparse
import warnings
from tools import EarlyStopping
from torch.utils.data import Dataset, DataLoader
from utils import get_data
from Model import ANN
from torch import nn
import torch
import copy
from tqdm import tqdm
import random 
import numpy as np
warnings.filterwarnings("ignore")



class Winkler_score():
    def __init__(self, alpha: float, reduction: str = 'none'):
        self.reduction = reduction
        self.alpha = alpha

    def score(self,l_interval,u_interval,y_t):
        score = u_interval - l_interval
        if y_t < l_interval:
            score += ((2/self.alpha) * (l_interval - y_t))
        elif y_t > u_interval:
            score += ((2/self.alpha) * (y_t - u_interval))
        return score

    def __call__(self, output: torch.Tensor, target: torch.Tensor):
        if self.alpha==0.2:
            u_interval_list = output[-1]
            l_interval_list = output[0]
        elif self.alpha==0.4:
            u_interval_list = output[-2]
            l_interval_list = output[1]
        elif self.alpha==0.6:
            u_interval_list = output[-3]
            l_interval_list = output[2]

        loss = torch.Tensor([
            self.score(l_interval, u_interval, y_t) 
            for l_interval, u_interval, y_t in zip(l_interval_list, u_interval_list, target)
        ])
        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss



class PinballLoss():
    def __init__(self, quantiles: torch.Tensor, reduction: str = 'none'):
        self.quantiles = quantiles
        self.reduction = reduction
        self.mse = nn.MSELoss()

    def __call__(self, output: torch.Tensor, target: torch.Tensor):
        loss = torch.zeros_like(target, dtype=torch.float)
        error = target - output  # * n*9

        first_term = self.quantiles * error
        second_term = (self.quantiles - 1) * error

        loss = torch.maximum(first_term, second_term)

        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss.mean()
            # loss = loss.mean() + 0.5 * self.mse(output.mean(-1), target.mean(-1))
        return loss



class EWCLoss(nn.Module):
    def __init__(self, model: nn.Module, dataset: Dataset, importance: float, quantiles):
        super(EWCLoss, self).__init__()
        self.model = model
        self.dataset = dataset
        self.importance = importance
        self.pre_params = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}  # Filter out the parameters that don't require gradients
        self.quantiles = quantiles
        self.device= next(self.model.parameters()).device
        self._precision_matrices = self._diag_fisher()
        print('self.importance',self.importance)

    def forward(self, output, target):
        loss = PinballLoss(self.quantiles,reduction='mean')(output, target)#.mean()
        for n, p in self.model.named_parameters():
            precision_matrix = self._precision_matrices[n]
            diff = (p - self.pre_params[n]) ** 2
            #print("precision_matrix:", precision_matrix)
            _loss = precision_matrix * diff
            #print("_loss:", _loss)
            loss += self.importance * _loss.sum()
        return loss

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in copy.deepcopy(self.pre_params).items():
            p.data.zero_()
            precision_matrices[n] = p.data

        self.model.eval()
        for idx, (seq_x,seq_x_concat,seq_y) in enumerate(self.dataset):
            self.model.zero_grad()
            seq_x = seq_x.to(self.device).float()
            seq_x_concat = seq_x_concat.to(self.device).float() 
            seq_y = seq_y.to(self.device).float()
            input_x=torch.cat((seq_x,seq_x_concat),1)
            output = self.model(input_x)
            loss = PinballLoss(self.quantiles)(output, seq_y).sum()
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: abs(p) for n, p in precision_matrices.items()}

        return precision_matrices



class evaluation():
    def __init__(self, args,quantiles) -> None:
        self.args = args
        self.quantiles = quantiles
        self.device = args.device
        self.model_type=self.args.model_type
        self.early_stop = EarlyStopping(patience=self.args.patience, verbose=False)
        self.quantiles=torch.tensor(quantiles).to(args.device)
        if self.args.forecasting_mode=='dot':
            self.criterion = nn.MSELoss()
        elif self.args.forecasting_mode=='prob':
            self.criterion = PinballLoss(self.quantiles, reduction='mean')

    def val(self,model, val_loader):
        model.eval()
        test_loss = 0
        all_outputs = []
        for idx, (seq_x, seq_x_concat, seq_y) in enumerate(val_loader):
            seq_x = seq_x.to(self.args.device).float()
            seq_x_concat = seq_x_concat.to(self.args.device).float()
            seq_y = seq_y.to(self.args.device).float()
            with torch.no_grad():
                if self.model_type=='NN':
                    input_x=torch.cat((seq_x,seq_x_concat),1)
                    output = model(input_x)
                else:
                    output = self.model(seq_x)
                loss = self.criterion(output, seq_y)
                test_loss += loss.item()
            all_outputs.append(output)
        all_outputs = torch.cat(all_outputs) # concatenate all outputs into a single tensor
        test_loss /= len(val_loader)
        return all_outputs, test_loss,model
    

class Trainer():
    def __init__(self, args_train, model, quantiles,mode='local',optimizer=None):
        self.model = model
        self.args = args_train
        self.device = args_train.device
        self.model_type=self.args.model_type
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.early_stop = EarlyStopping(patience=self.args.patience, verbose=False)
        self.quantiles=torch.tensor(quantiles).to(args_train.device)
        self.model.to(self.device)
        if  mode=='local':
            self.num_epochs = self.args.local_epochs
            self.early_stop_flag=False
        elif mode=='global' or mode=='central':
            self.num_epochs = self.args.global_epochs
            self.early_stop_flag=True
        elif mode=='fine tune':
            self.num_epochs = self.args.fine_tune_epochs
            self.early_stop_flag=True
        if self.args.forecasting_mode=='dot':
            self.criterion = nn.MSELoss()
        elif self.args.forecasting_mode=='prob':
            self.criterion = PinballLoss(self.quantiles, reduction='mean')


    def get_opt_info(self):
        params = self.optimizer.param_groups[0]
        learning_rate = params['lr']
        weight_decay = params['weight_decay']
        print(learning_rate,weight_decay)

    def train(self,train_loader,val_loader,test_loader,info='',ewc=False):
        if ewc:
            self.criterion=EWCLoss(self.model,val_loader,self.args.importance,self.quantiles)
        model_name=info+'.pth'
        self.evaluation=evaluation(self.args,self.quantiles)
        train_loss = 0.0
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        for epoch in range(self.num_epochs):
            self.model.train()
            for idx, (seq_x, seq_x_concat, seq_y) in enumerate(train_loader):
                seq_x = seq_x.to(self.args.device).float()
                seq_x_concat = seq_x_concat.to(self.args.device).float()
                seq_y = seq_y.to(self.args.device).float()
                self.optimizer.zero_grad()
                if self.model_type=='NN':
                    input_x=torch.cat((seq_x,seq_x_concat),1)
                    output = self.model(input_x)
                else:
                    output = self.model(seq_x)
                loss = self.criterion(output, seq_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            if not(self.early_stop_flag):
                print('Epoch: {} | Loss: {:.4f}'.format(epoch, loss.item()))
            else:
                self.model.eval()
                _,val_loss,_ = self.evaluation.val(self.model,val_loader)
                print(f"Epoch [{epoch+1}/{self.num_epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")
                early_stopping(val_loss, self.model,self.args.model_save_path,name=model_name)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
        if self.early_stop_flag:                
            self.model.load_state_dict(torch.load(self.args.model_save_path+model_name))
        else:
            return self.model
        
