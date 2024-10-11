import os
import numpy as np
import pandas as pd
from config import net_config
import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

import random
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')
class Dataset_train(Dataset):
    def __init__(self,root_path,train_list,signaldata_len,all_channels,size =None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.signaldata_len = signaldata_len
        assert self.signaldata_len > self.seq_len + self.pred_len,"单一数据长度必须满足输入数据长度加上预测数据长度"
        self.singleSignal_num = self.signaldata_len - self.seq_len - self.pred_len + 1 
        self.all_channels = all_channels
        self.upper_semg = [];self.lowwer_semg = [];self.acc = [];self.hands_pose = [];self.all_semg=[]
        # self.xbegin = xbegin;self.xend = xend;self.ybegin = ybegin;self.yend = yend
        self.data_len = 0  
        self.scaler = MinMaxScaler()
        self.__readdata__(root_path,train_list)
    def __readdata__(self,file_path,train_list):
        for i in train_list:
            upper_semgs,lowwer_semgs,accs,hands_poses = torch.load(os.path.join(file_path,"subject{0}.pt").format(i)).values()
            self.data_len += len(upper_semgs)
            for row in range(0,len(upper_semgs)):
                self.upper_semg.append(upper_semgs[row])
                self.lowwer_semg.append(lowwer_semgs[row])
                self.acc.append(self.add_gaussian_noise(accs[row]))
                self.hands_pose.append(self.add_gaussian_noise(hands_poses[row]))
                self.all_semg.append(torch.cat((upper_semgs[row],lowwer_semgs[row]),dim=1))
        self.upper_semg = torch.stack(self.upper_semg)
        self.lowwer_semg = torch.stack(self.lowwer_semg)
        self.acc = torch.stack(self.acc)
        self.hands_pose = torch.stack(self.hands_pose)
        self.all_semg= torch.stack(self.all_semg)
        #####2d
        self.all_semg_2d = self.all_semg.reshape(-1,self.all_semg.shape[-1])
        self.upper_semg_2d = self.upper_semg.reshape(-1,self.upper_semg.shape[-1])
        self.lowwer_semg_2d = self.lowwer_semg.reshape(-1,self.lowwer_semg.shape[-1])
        self.acc_2d = self.acc.reshape(-1,self.acc.shape[-1])
        self.hands_pose_2d = self.hands_pose.reshape(-1,self.hands_pose.shape[-1])

        if self.all_channels == True:
            self.scler.fit(self.all_semg_2d)
            self.all_semg = torch.from_numpy(self.scaler.transform(self.all_semg_2d)
                        .reshape(self.all_semg.shape[0],self.all_semg.shape[1],self.all_semg.shape[2])).to(torch.float32)
        else:
            self.scaler.fit(self.upper_semg_2d)
            self.upper_semg = torch.from_numpy(self.scaler.transform(self.upper_semg_2d)
                        .reshape(self.upper_semg.shape[0],self.upper_semg.shape[1],self.upper_semg.shape[2])).to(torch.float32)      
    def __getitem__(self, index):
        signal_num = index//self.singleSignal_num
        x_startindex = (index-1)%self.singleSignal_num
        x_endindex = x_startindex + self.seq_len
        y_startindex = x_endindex - self.label_len
        y_endindex = y_startindex + self.pred_len
        all_semg = self.all_semg[signal_num,x_startindex:x_endindex,:]
        upper_semg = self.upper_semg[signal_num,x_startindex:x_endindex,:]
        
        # lowwer_semg = self.lowwer_semg[signal_num,x_startindex:x_endindex,:]
        acc = self.acc[signal_num,x_startindex:x_endindex,:]
        hands_pose = self.hands_pose[signal_num,x_startindex:x_endindex,:]
        pose_max = torch.max(hands_pose,dim=0)
        # pose_avg = torch.mean(hands_pose,dim=0)
        # pose_es = torch.cat((pose_max.values,pose_avg),dim=0)
        pose_es = pose_max.values
        mask =torch.tensor(torch.cat((torch.ones(self.label_len),torch.zeros(self.pred_len)),dim=0),dtype=torch.bool)
        # all_semg_mask = torch.masked_select(self.all_semg[signal_num,y_startindex:y_endindex,:],mask)
        # upper_semg_mask = torch.masked_select(self.upper_semg[signal_num,y_startindex:y_endindex,:],mask)
        # lowwer_semg_mask = torch.masked_select(self.lowwer_semg[signal_num,y_startindex:y_endindex,:],mask)
        # acc_mask = torch.masked_select(self.acc[signal_num,y_startindex:y_endindex,:],mask)
        # hands_pose_mask = torch.masked_select(self.hands_pose[signal_num,y_startindex:y_endindex,:],mask)
        # all_semg_mask = torch.zeros_like(self.all_semg[signal_num,y_startindex:y_endindex,:])
        upper_semg_mask = torch.zeros_like(self.upper_semg[signal_num,y_startindex:y_endindex,:])
        # lowwer_semg_mask = torch.zeros_like(self.lowwer_semg[signal_num,y_startindex:y_endindex,:])
        acc_mask = torch.zeros_like(self.acc[signal_num,y_startindex:y_endindex,:])
        hands_pose_mask = torch.zeros_like(self.hands_pose[signal_num,y_startindex:y_endindex,:])
        mask_len = random.randint(0,5)
        return all_semg,upper_semg,acc,hands_pose,upper_semg_mask,acc_mask,hands_pose_mask,pose_es,mask_len
        
    def __len__(self):
        return self.data_len*self.singleSignal_num
    def add_gaussian_noise(self,tensor):
        std=net_config.std
        noise = torch.randn(tensor.size()) * std
        noisy_tensor = tensor + noise
        return noisy_tensor