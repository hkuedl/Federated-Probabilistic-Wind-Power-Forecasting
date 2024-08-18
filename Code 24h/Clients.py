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
from Train import Trainer, evaluation, PinballLoss
warnings.filterwarnings("ignore")

class Client():
    def __init__(self, args):
        self.args = args
        self.name=self.args.dataset_paths
        self.train_data, self.train_loader = get_data(self.args,flag='train')
        self.val_data, self.val_loader = get_data(self.args,flag='val')
        self.test_data, self.test_loader = get_data(self.args,flag='test')
        self.local_model=ANN(self.args).to(self.args.device)
        self.fed_local_model=ANN(self.args).to(self.args.device)
        self.local_trainer=Trainer(self.args,self.local_model,self.args.quantiles,mode='global')
        self.fed_local_trainer=Trainer(self.args,self.fed_local_model,self.args.quantiles,mode='local')
        self.evaluation=evaluation(self.args,self.args.quantiles)
        
    def local_train(self):
        self.local_trainer.train(self.train_loader,self.val_loader,self.test_loader,info=self.name)
        return self.local_model
    
    def fed_local_train(self):
        self.fed_local_trainer.train(self.train_loader,self.val_loader,self.test_loader,info=self.name)
        return self.fed_local_model
    
    def local_evaluation(self):
        return self.evaluation.val(self.local_model,self.test_loader)
        
    def fed_local_evaluation(self):
        return self.evaluation.val(self.fed_local_model,self.test_loader)
    
    def set_fed_local_model(self,global_parameters):
        self.fed_local_model.load_state_dict(global_parameters)
    
    def get_fed_local_model(self):
        return self.fed_local_model
    
    def get_local_model(self):
        return self.local_model
    
    def get_dataset(self,flag='train'):
        if flag=='train':
            return self.train_data, self.train_loader
        elif flag=='val':
            return self.val_data, self.val_loader
        elif flag=='test':
            return self.test_data, self.test_loader
    

    def local_fine_tune(self,ewc_flag=False,importance=0.1,fine_tune_epochs=20,fine_tune_lr=1e-5,patience=3):
        self.fined_local_model=ANN(self.args).to(self.args.device)
        self.fined_local_model.load_state_dict(self.fed_local_model.state_dict())  
        args_temp=copy.deepcopy(self.args)
        args_temp.lr=fine_tune_lr
        args_temp.importance=importance
        args_temp.fine_tune_epochs=fine_tune_epochs
        args_temp.patience=patience
        self.fine_tune_trainer=Trainer(args_temp,self.fined_local_model,self.args.quantiles,mode='fine tune')
        self.fine_tune_trainer.get_opt_info()
        self.fine_tune_trainer.train(self.train_loader,self.val_loader,self.test_loader,info=self.name,ewc=ewc_flag)
        return self.evaluation.val(self.fined_local_model,self.test_loader)

    def copy_client(self,client):
        self.local_model=copy.deepcopy(client.local_model)
        self.fed_local_model=copy.deepcopy(self.fed_local_model)
        self.train_data, self.train_loader = copy.deepcopy(client.train_data) ,copy.deepcopy(client.train_loader) 
        self.val_data, self.val_loader = copy.deepcopy(client.val_data) ,copy.deepcopy(client.val_loader)
        self.test_data, self.test_loader = copy.deepcopy(client.test_data) ,copy.deepcopy(client.test_loader)
        self.local_trainer=copy.deepcopy(client.local_trainer)
        self.fed_local_trainer=copy.deepcopy(client.fed_local_trainer)
        self.evaluation=copy.deepcopy(client.evaluation)

    
class Client_mul():
    def __init__(self, args):
        self.args = args
        self.name=self.args.dataset_paths
        self.train_data, self.train_loader = get_data(self.args,flag='train')
        self.val_data, self.val_loader = get_data(self.args,flag='val')
        self.test_data, self.test_loader = get_data(self.args,flag='test')
        self.local_model=ANN(self.args).to(self.args.device)
        self.fed_local_models = [ANN(self.args).to(self.args.device) for _ in range(self.args.k)]
        self.fed_local_trainers=[Trainer(self.args,self.fed_local_models[i],self.args.quantiles,mode='local') for i in range(self.args.k)]
        self.local_trainer=Trainer(self.args,self.local_model,self.args.quantiles,mode='global')
        self.evaluation=evaluation(self.args,self.args.quantiles)
        self.model_score=[0 for i in range(self.args.k)]
        
    def local_train(self,ewc=False):
        print(self.name)
        self.local_trainer.train(self.train_loader,self.val_loader,self.test_loader,info=self.name,ewc=ewc)
        return self.local_model
    
    def fed_local_train(self,ewc=False):
        print('ewc:',ewc)
        for i in range(self.args.k):
            print(self.name+' model: '+str(i))
            self.fed_local_trainers[i].train(self.train_loader,self.val_loader,self.test_loader,info=self.name,ewc=ewc)
        return self.fed_local_models
    
    def local_evaluation(self):
        return self.evaluation.val(self.local_model,self.test_loader)
        
    def fed_local_evaluation(self,dataset_type='test'):
        outputs,test_losses,models=[],[],[]   
        for i in range(self.args.k):
            if dataset_type=='val':
                output,test_loss,model=self.evaluation.val(self.fed_local_models[i],self.val_loader)
            else:
                output,test_loss,model=self.evaluation.val(self.fed_local_models[i],self.test_loader)
            outputs.append(output)
            test_losses.append(test_loss)
            models.append(model)
        return outputs,test_losses,models
    
    def set_fed_local_model(self,global_parameters):
        for i in range(self.args.k):
            self.fed_local_models[i].load_state_dict(global_parameters[i])
    
    def get_fed_local_model(self):
        return self.fed_local_models
    
    def get_local_model(self):
        return self.local_model

    def generates_mask(self):
        _, test_losses, _ = self.fed_local_evaluation(dataset_type='val')
        min_loss_index = min(range(len(test_losses)), key=test_losses.__getitem__)
        self.model_score[min_loss_index]=self.model_score[min_loss_index]*self.args.decay+1
        print(test_losses,min_loss_index)
        print(self.model_score)
        return np.argmax(self.model_score)
        
    
    def get_dataset(self,flag='train'):
        if flag=='train':
            return self.train_data, self.train_loader
        elif flag=='val':
            return self.val_data, self.val_loader
        elif flag=='test':
            return self.test_data, self.test_loader
        
    def local_fine_tune(self,model_index,ewc_flag=False,importance=0.1,fine_tune_epochs=20,patience=3,fine_tune_lr=1e-5):
        self.fined_local_model=ANN(self.args).to(self.args.device)
        self.fined_local_model.load_state_dict(self.fed_local_models[model_index].state_dict())  
        args_temp=copy.deepcopy(self.args)
        args_temp.lr=fine_tune_lr
        args_temp.importance=importance
        args_temp.fine_tune_epochs=fine_tune_epochs
        args_temp.patience=patience
        self.fine_tune_trainer=Trainer(args_temp,self.fined_local_model,self.args.quantiles,mode='fine tune')
        self.fine_tune_trainer.get_opt_info()
        self.fine_tune_trainer.train(self.train_loader,self.val_loader,self.test_loader,info=self.name,ewc=ewc_flag)
        return self.evaluation.val(self.fined_local_model,self.test_loader)
    

    def copy_client(self,client):
        self.local_model=copy.deepcopy(client.local_model)
        if hasattr(client, 'fed_local_models'):
            self.fed_local_models=copy.deepcopy(client.fed_local_models)
            self.fed_local_trainers=copy.deepcopy(client.fed_local_trainers)
        self.train_data = copy.deepcopy(client.train_data)
        self.val_data = copy.deepcopy(client.val_data)
        self.test_data = copy.deepcopy(client.test_data)
        self.train_loader = copy.deepcopy(client.train_loader)
        self.val_loader = copy.deepcopy(client.val_loader)
        self.test_loader = copy.deepcopy(client.test_loader)
        self.local_trainer=copy.deepcopy(client.local_trainer)
        self.evaluation=copy.deepcopy(client.evaluation)