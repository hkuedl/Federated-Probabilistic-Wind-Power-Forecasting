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
from Train import Trainer, evaluation, PinballLoss
from tqdm import tqdm
import random 
import numpy as np
from Clients import Client
warnings.filterwarnings("ignore")


class Server():
    def __init__(self, args,clients):
        self.args = args
        self.global_model=ANN(self.args).to(self.args.device)
        self.clients=clients
        self.evaluation=evaluation(self.args,self.args.quantiles)
        self.quantiles=torch.tensor(self.args.quantiles).to(args.device)
        if self.args.forecasting_mode=='dot':
            self.criterion = nn.MSELoss()
        elif self.args.forecasting_mode=='prob':
            self.criterion = PinballLoss(self.quantiles, reduction='mean')

        self.global_train_dataset, self.global_train_loader = get_data(self.args,flag='train')
        self.global_val_dataset, self.global_val_loader = get_data(self.args,flag='val')
        self.global_test_dataset, self.global_test_loader = get_data(self.args,flag='test')

    def fed_train(self):
        model_name='fed.pth'
        path = self.args.model_save_path
        early_stopping=EarlyStopping(patience=self.args.fed_patience, verbose=False)
        initial_params = self.global_model.state_dict()
        for client in self.clients:
            client.set_fed_local_model(initial_params)

        # Local train
        for epoch in range(self.args.global_epochs):

            fed_local_losses=[]
            fed_local_preds=[]
            fed_local_models=[]
            for i in range(self.args.number_clients):
                fed_local_pred,fed_local_loss,fed_local_model=self.clients[i].fed_local_evaluation()
                fed_local_losses.append(fed_local_loss)
                fed_local_preds.append(fed_local_pred)
                fed_local_models.append(fed_local_model)
            print('test performance:',fed_local_losses)

            self.global_model.train()
            for client in self.clients:
                client.fed_local_train()

            global_model_params = self.model_average()
            self.global_model.load_state_dict(global_model_params)

            for client in self.clients:
                client.set_fed_local_model(global_model_params)


            self.global_model.eval()
            _,val_loss,_ = self.evaluation.val(self.global_model,self.global_val_loader)
            print(f"Federated training Epoch [{epoch+1}/{self.args.global_epochs}] Val Loss: {val_loss:.4f}")
            
            early_stopping(val_loss, self.global_model ,path,name=model_name)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        self.global_model.load_state_dict(torch.load(self.args.model_save_path+model_name))
        return self.global_model
    
    def model_average(self):
        local_models = []
        for client in self.clients:
            model = client.get_fed_local_model()
            local_models.append(model)
        
        avg_model_params = local_models[0].state_dict()

        for param_name in avg_model_params:
            for i in range(1, len(local_models)):
                avg_model_params[param_name] += local_models[i].state_dict()[param_name]
            avg_model_params[param_name] /= len(local_models)
        
        return avg_model_params

    def central_train(self):
        self.central_model=ANN(self.args).to(self.args.device)
        self.central_trainer=Trainer(self.args,self.central_model,self.args.quantiles,mode='central')
        self.central_trainer.train(self.global_train_loader,self.global_val_loader,self.global_test_loader,info='central')
        return self.central_model
    
    def central_evaluation(self,dataset=-1):
        if dataset==-1:
            return self.evaluation.val(self.central_model,self.global_test_loader)
        if dataset in range(len(self.clients)):
            _,loader=self.clients[dataset].get_dataset(flag='test')
            return self.evaluation.val(self.central_model,loader)
    
    def global_evaluation(self,dataset=-1):
        if dataset==-1:
            return self.evaluation.val(self.global_model,self.global_test_loader)
        if dataset in range(len(self.clients)):
            _,loader=self.clients[dataset].get_dataset(flag='test')
            return self.evaluation.val(self.global_model,loader)

    def local_train(self):
        print('Launch Local Training!')
        for client in self.clients:
            client.local_train()


    def copy_server(self,server):
        self.global_model=copy.deepcopy(server.global_model)
        self.clients=copy.deepcopy(server.clients)
        self.central_model=copy.deepcopy(server.central_model)
        self.evaluation=copy.deepcopy(server.evaluation)
        self.quantiles=copy.deepcopy(server.quantiles)
        self.criterion=copy.deepcopy(server.criterion)
        self.global_train_dataset=copy.deepcopy(server.global_train_dataset)
        self.global_train_loader=copy.deepcopy(server.global_train_loader)
        self.global_val_dataset=copy.deepcopy(server.global_val_dataset)
        self.global_val_loader=copy.deepcopy(server.global_val_loader)
        self.global_test_dataset=copy.deepcopy(server.global_test_dataset)
        self.global_test_loader=copy.deepcopy(server.global_test_loader)
        self.central_trainer=copy.deepcopy(server.central_trainer)
        self.args=copy.deepcopy(server.args)

class Server_mul():
    def __init__(self, args,clients):
        self.args = args
        self.clients=clients
        self.warm_up_epochs=self.args.warm_up_epochs
        self.selection_epochs=self.args.selection_epochs  
        self.global_models=[ANN(self.args).to(self.args.device) for _ in range(self.args.k)]
        self.index_set=[0 for _ in range(len(self.clients))]
        self.evaluation=evaluation(self.args,self.args.quantiles)
        self.quantiles=torch.tensor(self.args.quantiles).to(args.device)
        if self.args.forecasting_mode=='dot':
            self.criterion = nn.MSELoss()
        elif self.args.forecasting_mode=='prob':
            self.criterion = PinballLoss(self.quantiles, reduction='mean')

        self.global_train_dataset, self.global_train_loader = get_data(self.args,flag='train')
        self.global_val_dataset, self.global_val_loader = get_data(self.args,flag='val')
        self.global_test_dataset, self.global_test_loader = get_data(self.args,flag='test')

    def fed_train(self):
        model_name_lst=['fed'+str(i)+'.pth' for i in range(self.args.k)]
        self.stop_flag=[False for _ in range(self.args.k)]
        self.selection_stop_flag=False
        self.index_set_buffer=[]
        self.early_stop_start=False
        path = self.args.model_save_path
        early_stopping_lst=[EarlyStopping(patience=self.args.fed_patience, verbose=False) for _ in range(self.args.k)] 
        initial_params = [self.global_models[i].state_dict() for i in range(self.args.k)]
        for client in self.clients:
            client.set_fed_local_model(initial_params)

        # Local train
        for epoch in range(self.args.global_epochs):
            fed_local_losses=[]
            fed_local_preds=[]
            fed_local_models=[]
            for i in range(self.args.number_clients):
                fed_local_pred,fed_local_loss,fed_local_model=self.clients[i].fed_local_evaluation()
                fed_local_losses.append(fed_local_loss[self.index_set[i]])
                fed_local_preds.append(fed_local_pred[self.index_set[i]])
                fed_local_models.append(fed_local_model[self.index_set[i]])
            print('test performance:',fed_local_losses)

            print(self.selection_stop_flag,self.stop_flag)
            for client_id in range(len(self.clients)):
                if epoch>=self.warm_up_epochs and self.selection_stop_flag==True and self.stop_flag[self.index_set[client_id]]==True:
                    pass
                elif epoch>=self.warm_up_epochs and self.selection_stop_flag==True and self.stop_flag[self.index_set[client_id]]==False:
                    self.clients[client_id].fed_local_train(ewc=True) #importance=0,等于没开启ewc
                else:
                    self.clients[client_id].fed_local_train()
            if epoch<self.warm_up_epochs:
                global_model_params_lst = self.model_average(use_mask=False)
            else:
                global_model_params_lst = self.model_average(use_mask=True) 
                self.selection_stop_flag=self.update_index_buffer()
                print(self.index_set_buffer)
            for i in range(self.args.k):
                if self.stop_flag[i]==False:
                    self.global_models[i].load_state_dict(global_model_params_lst[i])

            for client_id in range(len(self.clients)):
                self.clients[client_id].set_fed_local_model([self.global_models[i].state_dict() for i in range(self.args.k)])

            val_loss=self.fed_evaluation()
            for i in range(self.args.k):
                print(f"Federated training Epoch [{epoch+1}/{self.args.global_epochs}] Val Loss: {val_loss[i]:.4f}")
                if self.selection_stop_flag:
                    if self.early_stop_start:
                        if i not in self.index_set:
                            self.stop_flag[i]=True
                        early_stopping_lst[i](val_loss[i], self.global_models[i],path,name=model_name_lst[i])
                        if early_stopping_lst[i].early_stop:
                            print("model Early stopping: ",i)
                            self.stop_flag[i]=True
                    self.early_stop_start=True
            if self.stop_flag==[True for _ in range(self.args.k)]:
                break
            

        for i in range(self.args.k):
            self.global_models[i].load_state_dict(torch.load(path+model_name_lst[i]))
        return self.global_models
        
    def update_index_buffer(self):
        self.index_set_buffer.append(copy.deepcopy(self.index_set))
        if len(self.index_set_buffer)<self.selection_epochs:
            return False
        else:
            temp_flag=True
            for i in range(len(self.index_set_buffer)-1):
                self.index_set_buffer[i]!=self.index_set
                temp_flag=False
                break
            if temp_flag:
                return True
            else:
                self.index_set_buffer=self.index_set_buffer[-self.selection_epochs:]
                if len(set(tuple(x) for x in self.index_set_buffer)) == 1:
                    return True
                else:
                    return False
    
    def model_average(self, use_mask=False):
        local_prefered_lists = [[] for i in range(self.args.k)]
        local_all_lists = [[] for i in range(self.args.k)]
        avg_model_params_lists = []
        for client_id in range(len(self.clients)):
            local_model_list = self.clients[client_id].get_fed_local_model()
            for i in range(self.args.k):
                local_all_lists[i].append(local_model_list[i])
            if use_mask:
                if self.selection_stop_flag:
                    index = self.index_set[client_id]
                else:
                    index = self.clients[client_id].generates_mask()
                    self.index_set[client_id] = index
                local_prefered_lists[index].append(local_model_list[index])

        if use_mask:
            print('model selection result:')
            print(self.index_set)

        for i in range(self.args.k):
            models_for_avg = local_prefered_lists[i] if use_mask and len(local_prefered_lists[i]) > 0 else local_all_lists[i]
            avg_model_params = models_for_avg[0].state_dict()

            for param_name in avg_model_params:
                for i in range(1, len(models_for_avg)):
                    avg_model_params[param_name] += models_for_avg[i].state_dict()[param_name]
                avg_model_params[param_name] /= len(models_for_avg)
            avg_model_params_lists.append(avg_model_params)

        return avg_model_params_lists

    def fed_evaluation(self):
        losses=[[] for _ in range(self.args.k)]
        for client_id in range(len(self.clients)):
            index=self.index_set[client_id]
            _,test_loss,_=self.clients[client_id].fed_local_evaluation(dataset_type='val')
            print(f'Client {client_id} test loss: {test_loss}')
            losses[index].append(test_loss[index])
        avg_loss=[sum(i)/len(i) if len(i)!=0 else 10000 for i in losses]
        return avg_loss

    def central_train(self):
        self.central_model=ANN(self.args).to(self.args.device)
        self.central_trainer=Trainer(self.args,self.central_model,self.args.quantiles,mode='central')
        self.central_trainer.train(self.global_train_loader,self.global_val_loader,self.global_test_loader,info='central')
        return self.central_model
    
    def central_evaluation(self,dataset=-1):
        if dataset==-1:
            return self.evaluation.val(self.central_model,self.global_test_loader)
        if dataset in range(len(self.clients)):
            _,loader=self.clients[dataset].get_dataset(flag='test')
            return self.evaluation.val(self.central_model,loader)
    
    def global_evaluation(self,dataset=-1):
        if dataset==-1:
            return self.evaluation.val(self.global_model,self.global_test_loader)
        if dataset in range(len(self.clients)):
            _,loader=self.clients[dataset].get_dataset(flag='test')
            return self.evaluation.val(self.global_model,loader)

    def local_train(self):
        print('Launch Local Training!')
        for client in self.clients:
            client.local_train()

    def copy_server(self,server):
        self.clients=copy.deepcopy(server.clients)
        self.global_models=copy.deepcopy(server.global_models)
        self.central_model=copy.deepcopy(server.central_model)
        self.index_set=copy.deepcopy(server.index_set)
        self.index_set_buffer=copy.deepcopy(server.index_set_buffer)
        self.stop_flag=copy.deepcopy(server.stop_flag)
        self.selection_stop_flag=copy.deepcopy(server.selection_stop_flag)
        self.evaluation=copy.deepcopy(server.evaluation)
        self.quantiles=copy.deepcopy(server.quantiles)
        self.criterion=copy.deepcopy(server.criterion)
        self.global_train_dataset=copy.deepcopy(server.global_train_dataset)
        self.global_train_loader=copy.deepcopy(server.global_train_loader)
        self.global_val_dataset=copy.deepcopy(server.global_val_dataset)
        self.global_val_loader=copy.deepcopy(server.global_val_loader)
        self.global_test_dataset=copy.deepcopy(server.global_test_dataset)
        self.global_test_loader=copy.deepcopy(server.global_test_loader)
        self.central_trainer=copy.deepcopy(server.central_trainer)
        self.args=copy.deepcopy(server.args)
    
