from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
class Dataset_Custom(Dataset):
    def __init__(self,  root_path='../Data/GFC12/', flag='train', size=[96,0,24], train_length=16800,
                data_path='wf1.csv', target='target', scale=True, inverse=True):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.root_path = root_path
        self.data_path = data_path
        #self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.flag = flag
        self.train_length = train_length
        if isinstance(self.data_path, list):
            self.X_nor, self.y_nor, self.X_concat_nor = self.process_datasets(self.data_path)
        else:
            self.X_nor, self.y_nor, self.X = self.__read_data__(self.data_path)

    def __read_data__(self,data_path):
        self.scaler_x = MinMaxScaler()# StandardScaler()
        self.scaler_y = MinMaxScaler()# StandardScaler()
        self.scaler_x_concat = MinMaxScaler()# StandardScaler()
        file_name=data_path
        wind_data=pd.read_csv(self.root_path+data_path+'.csv')
        wind_data = wind_data.interpolate(method='cubic', limit_direction='both')

        wind_data[self.target]=wind_data[file_name].shift(-self.pred_len)
        wind_data['target_date']=wind_data['date'].shift(-self.pred_len)
        wind_data['target_date'] = pd.to_datetime(wind_data['target_date'])
        wind_data['target_month']=wind_data['target_date'].dt.month
        wind_data['target_week']=wind_data['target_date'].dt.week
        wind_data['target_hour']=wind_data['target_date'].dt.hour
        wind_data['pred_u']=wind_data['u'].shift(-self.pred_len)
        wind_data['pred_v']=wind_data['v'].shift(-self.pred_len)
        del wind_data['hors']

        wind_data.index=range(len(wind_data))

        for i in range(1,self.seq_len):
            wind_data['u_'+str(i)]=wind_data['u'].shift(i)

        for i in range(1,self.seq_len):
            wind_data['v_'+str(i)]=wind_data['v'].shift(i)

        for i in range(0,self.seq_len):
            wind_data['wf_'+str(i)]=wind_data[file_name].shift(i)

        u_features = ['u_'+str(i) if i != 0 else 'u' for i in range(self.seq_len)]
        v_features = ['v_'+str(i) if i != 0 else 'v' for i in range(self.seq_len)]
        wf_features = ['wf_'+str(i) for i in range(self.seq_len)]
        self.features = u_features + v_features + wf_features
        self.features_concat=['target_month','target_week','target_hour','pred_u','pred_v']
        cols=['date']+u_features+v_features+wf_features+self.features_concat+[file_name,self.target]
        wind_data = wind_data.reindex(columns=cols)
        wind_data.dropna(inplace=True)

        # 划分训练和测试数据
        train_data = wind_data[0:self.train_length]
        test_data = wind_data[self.train_length:]


        # 提取训练数据的特征和目标
        X_train_before_split = train_data[self.features]
        X_concat_train_before_split = train_data[self.features_concat]
        y_train_before_split = train_data[self.target]

        X_test = test_data[self.features]
        X_concat_test = test_data[self.features_concat]
        y_test = test_data[self.target]

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(X_train_before_split, y_train_before_split, test_size=0.2, random_state=42)
        X_concat_train, X_concat_val, _, _ = train_test_split(X_concat_train_before_split, y_train_before_split, test_size=0.2, random_state=42)
        
        if self.flag == 'train':
            self.X = X_train
            self.y = y_train
            self.X_concat = X_concat_train
        elif self.flag == 'val':
            self.X = X_val
            self.y = y_val
            self.X_concat = X_concat_val
        else:
            self.X  = X_test
            self.y = y_test
            self.X_concat = X_concat_test

        if self.scale:
            self.scaler_x.fit(X_train)
            self.scaler_y.fit(y_train.values.reshape(-1, 1))
            self.scaler_x_concat.fit(X_concat_train)

            self.X_nor=self.scaler_x.transform(self.X.values)
            self.y_nor=self.scaler_y.transform(self.y.values.reshape(-1, 1))
            self.X_concat_nor=self.scaler_x_concat.transform(self.X_concat.values)
        return self.X_nor, self.y_nor, self.X_concat_nor

    def process_datasets(self, path_lst):
        X_train_all = None
        y_train_all = None
        X_concat_train_all = None

        for path in path_lst:
            sub_X, sub_y, sub_X_concat = self.__read_data__(path)
            if X_train_all is None:
                X_train_all = sub_X
                y_train_all = sub_y
                X_concat_train_all = sub_X_concat
            else:
                X_train_all = np.concatenate([X_train_all, sub_X])
                y_train_all = np.concatenate([y_train_all, sub_y])
                X_concat_train_all = np.concatenate([X_concat_train_all, sub_X_concat])

        return X_train_all, y_train_all, X_concat_train_all


    def __getitem__(self, index):
        seq_x=self.X_nor[index]
        seq_y=self.y_nor[index]
        seq_x_concat=self.X_concat_nor[index]    
        return seq_x, seq_x_concat,seq_y

    def __len__(self):
        return len(self.X)

    def inverse_transform(self, y_nor):
        return self.scaler_y.inverse_transform(y_nor)