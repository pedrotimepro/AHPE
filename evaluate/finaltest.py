# encoding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 
import random
import time
import pyvista as pv
import torch.utils
from config import  net_config
from utils.metrics import metric
from dataloader import Dataset_train
import DoubleDecoder
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from torch import optim
import os   
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from math import pi
from numpy import deg2rad
import pyvista as pv
import torch
from trimesh import Trimesh
from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer, MANOOutput
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import time
import torch.nn.functional as F
import torch.nn as nn
def movehands(R_CMC1_F,R_CMC1_A,R_MCP1_F,R_MCP2_F,R_MCP3_F,R_MCP4_F,R_MCP4_A,R_MCP5_F,R_MCP5_A,R_IP1_F,R_PIP2_F,R_PIP3_F,R_PIP4_F,R_PIP5_F,R_DIP2,R_DIP3,R_DIP4,R_DIP5,wrist_F,wrist_A):
    mano_layer = ManoLayer(rot_mode="axisang",
                           center_idx=9,
                           mano_assets_root="assets/mano",
                           use_pca=False,
                           flat_hand_mean=True)
    hand_faces = mano_layer.th_faces  # (NF, 3)

    axisFK = AxisLayerFK(mano_assets_root="assets/mano")
    composed_ee = torch.zeros((1, 16, 3))
    cmc1_F = deg2rad(R_CMC1_F);cmc1_A = deg2rad(R_CMC1_A);MCP1_F = deg2rad(R_MCP1_F);pip1_F = deg2rad(R_IP1_F)
    MCP2_F =deg2rad(R_MCP2_F);PIP2_F =deg2rad(R_PIP2_F);DIP2_F =deg2rad(R_DIP2)
    MCP3_F =deg2rad(R_MCP3_F);PIP3_F = deg2rad(R_PIP3_F);DIP3 =deg2rad(R_DIP3)
    MCP4_F =deg2rad(R_MCP4_F);MCP4_A =deg2rad(R_MCP4_A);PIP4_F =deg2rad(R_PIP4_F);DIP4 =deg2rad(R_DIP4)
    MCP5_F =deg2rad(R_MCP5_F);MCP5_A =deg2rad(R_MCP5_A);PIP5_F =deg2rad(R_PIP5_F);DIP5 =deg2rad(R_DIP5)
    wrist_A = deg2rad(wrist_A);wrist_F = deg2rad(wrist_F)
    #415 修正 按照论文进行修正
    cmc1f_error = 25 ;cmc1_A_error = 0;pip1_error = 25;mcp1_error =40
    mcp2_error =38 ; pip2_error =35;pip2A_error= 2; mcp2A_error =-20
    mcp3_error =28 ; pip3_error =25;mcp3A_error = -3
    mcp4_error =26 ; pip4_error =35;MCP4A_error =10
    mcp5_error =25; pip5_error =25; pip5_A_error =-10 ;MCP5_A_error = 30
    # 对模型和数据进行修正
    composed_ee[:,0] = torch.tensor([0,wrist_A,wrist_F])
    composed_ee[:,13] = torch.tensor([0,(cmc1_A+deg2rad(cmc1_A_error)),(cmc1_F+deg2rad(cmc1f_error))])
    composed_ee[:,14] = torch.tensor([0,0,(MCP1_F+deg2rad(mcp1_error))])
    composed_ee[:,15] = torch.tensor([0,0,pip1_F+np.deg2rad(pip1_error)])
    composed_ee[:,1] = torch.tensor([0,deg2rad(mcp2A_error),(MCP2_F+np.deg2rad(mcp2_error))])
    composed_ee[:,2] = torch.tensor([0,0,(PIP2_F+np.deg2rad(pip2_error))])
    composed_ee[:,3] = torch.tensor([0,0,DIP2_F])
    composed_ee[:,4] = torch.tensor([0,deg2rad(mcp3A_error),(MCP3_F+np.deg2rad(mcp3_error))])
    composed_ee[:,5] = torch.tensor([0,0,(PIP3_F+np.deg2rad(pip3_error))])
    composed_ee[:,6] = torch.tensor([0,0,DIP3])
    composed_ee[:,7] = torch.tensor([0,MCP5_A+np.deg2rad(MCP5_A_error),(MCP5_F+deg2rad(mcp5_error))])
    composed_ee[:,8] = torch.tensor([0,0,(PIP5_F+np.deg2rad(pip5_error))])
    composed_ee[:,9] = torch.tensor([0,0,DIP5])
    composed_ee[:,10] = torch.tensor([0,MCP4_A+deg2rad(MCP4A_error),(MCP4_F+deg2rad(mcp4_error))])
    composed_ee[:,11] = torch.tensor([0,0,PIP4_F+deg2rad(pip4_error)])
    composed_ee[:,12] = torch.tensor([0,0,DIP4])
    composed_aa = axisFK.compose(composed_ee).clone()  # (B=1, 16, 3)
    composed_aa = composed_aa.reshape(1, -1)  # (1, 16x3)
    zero_shape = torch.zeros((1, 10))

    mano_output: MANOOutput = mano_layer(composed_aa, zero_shape)
    T_g_p = mano_output.transforms_abs  # (B=1, 16, 4, 4)
    T_g_a, R, ee = axisFK(T_g_p)
    T_g_a = T_g_a.squeeze(0)
    hand_verts = mano_output.verts.squeeze(0)  # (NV, 3)
    hand_faces = mano_layer.th_faces  # (NF, 3)
    return hand_verts, hand_faces
def draw_compare(preds,trues,clims,name):
    pl = pv.Plotter(off_screen=False, polygon_smoothing=True)
    pl.set_background('white')
    pl.add_camera_orientation_widget()
    hand_verts_preds, hand_faces_preds = movehands(preds[0],preds[1],preds[2],preds[3],preds[4],preds[5],preds[6],preds[7],preds[8],preds[9],preds[10],preds[11],preds[12],preds[13],preds[14]
                                                   ,preds[15],preds[16],preds[17],preds[18],preds[19])
    hand_verts_ori,hand_faces_ori = movehands(trues[0],trues[1],trues[2],trues[3],trues[4],trues[5],trues[6],trues[7],trues[8],trues[9],trues[10],trues[11],trues[12],trues[13],trues[14]
                                              ,trues[15],trues[16],trues[17],trues[18],trues[19])
    mesh = pv.wrap(Trimesh(vertices=hand_verts_preds, faces=hand_faces_preds))
    mesh_ori = pv.wrap(Trimesh(vertices=hand_verts_ori, faces=hand_faces_ori))
    error_mesh_point = mesh.points - mesh_ori.points
    mesh['error'] = error_mesh_point
    sargs = dict(height=0.36, vertical=True, position_x=0.05, position_y=0.05)
    pl.add_mesh(mesh,cmap = "RdYlGn_r",clim=clims,scalars = 'error',scalar_bar_args=sargs)
    pl.view_isometric()
    pl.save_graphic("./result/"+name)
    pl.close()
class Dataset_all(Dataset):
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
### 评估参数
def get_test_result():
    all_list = net_config.shuffle_list
    device = torch.device("cpu")
    all_dataset = Dataset_all(net_config.data_path,all_list,
        net_config.signaldata_len,net_config.all_channels,size=net_config.size)
    all_loader = torch.utils.data.DataLoader(all_dataset,batch_size=1,shuffle=True)
    net = DoubleDecoder.Model(net_config).to(device)
    all_steps = len(all_dataset)
    net.load_state_dict(torch.load(net_config.model_path))
    net.eval()
    rms_loss = [];mae_loss = [];delta_median_loss = [];delta_mean_loss = [];delta_rr_loss = [];delta_cc_loss = []
    for i,(all_semg,upper_semg,acc,hands_pose,upper_semg_mask,acc_mask,hands_pose_mask,pose_es,mask_len) in  enumerate(all_loader):
        if i % 100 == 0:
            batch_x_mark = None
            batch_y_mark = None 
            batch_x  = upper_semg.to(device,dtype=torch.float32)
            batch_y = hands_pose.to(device,dtype =torch.float32)
            pose_es = pose_es.float().to(device)
            # decoder input
            pred_len = net_config.pred_len - mask_len[0]
            label_len = net_config.label_len + mask_len[0]
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
            outputs,pose_es_pred = net(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if net_config.features == 'MS' else 0
            outputs = outputs[:, -pred_len:, f_dim:]
            batch_y = batch_y[:, -pred_len:, f_dim:].to(device)
            #输出值与真实值的对比 采用 rms，mae,中值，均值，r2，CC
            rms = torch.sqrt(torch.mean(torch.pow(outputs - batch_y, 2)))
            mae = torch.mean(torch.abs(outputs - batch_y))
            outputs_median = torch.median(outputs)
            batch_y_median = torch.median(batch_y)
            delta_median = torch.abs(outputs_median - batch_y_median)
            outputs_mean = torch.mean(outputs)
            batch_y_mean = torch.mean(batch_y)
            delta_mean = torch.abs(outputs_mean - batch_y_mean)
            r2 = 1 - torch.sum(torch.pow(outputs - batch_y, 2)) / torch.sum(torch.pow(batch_y - torch.mean(batch_y), 2))
            x = outputs.flatten()
            y = batch_y.flatten()
            cc = (torch.sum((x - torch.mean(x)) * (y - torch.mean(y))) / torch.sqrt(torch.sum((x - torch.mean(x)) ** 2) * torch.sum((y - torch.mean(y)) ** 2)))
            print("step: %d, rms: %.4f, mae: %.4f, delta_median: %.4f, delta_mean: %.4f, r2: %.4f, cc: %.4f" % (i,rms.item(),mae.item(),delta_median.item(),delta_mean.item(),r2.item(),cc.item()))
            rms_loss.append(rms.item());mae_loss.append(mae.item());delta_median_loss.append(delta_median.item());delta_mean_loss.append(delta_mean.item());delta_rr_loss.append(r2.item());delta_cc_loss.append(cc.item())
    _rms = np.mean(rms_loss);_mae = np.mean(mae_loss);_delta_median = np.mean(delta_median_loss);_delta_mean = np.mean(delta_mean_loss);_delta_rr = np.mean(delta_rr_loss);_delta_cc = np.mean(delta_cc_loss)
    print('RMS: %.4f, MAE: %.4f, delta_median: %.4f, delta_mean: %.4f, delta_rr: %.4f, delta_cc: %.4f' % (_rms,_mae,_delta_median,_delta_mean,_delta_rr,_delta_cc))
    result = np.array([_rms,_mae,_delta_median,_delta_mean,_delta_rr,_delta_cc])
    np.save("./result/result_"+".npy",result)

## 绘制误差图像 按数据集计算不考虑预测的作为输入再预测
def draw_error_picture():
    all_list = net_config.shuffle_list
    device = torch.device("cpu")
    all_dataset = Dataset_all(net_config.data_path,all_list,
        net_config.signaldata_len,net_config.all_channels,size=net_config.size)
    all_loader = torch.utils.data.DataLoader(all_dataset,batch_size=1,shuffle=False)
    net = DoubleDecoder.Model(net_config).to(device)
    all_steps = len(all_dataset)
    net.load_state_dict(torch.load(net_config.model_path))
    net.eval()
    draw_i =random.randint(0,all_steps-1)
    draw_i = 10
    for i,(all_semg,upper_semg,acc,hands_pose,upper_semg_mask,acc_mask,hands_pose_mask,pose_es,mask_len) in  enumerate(all_loader):
        if i == draw_i:
            batch_x_mark = None
            batch_y_mark = None 
            batch_x  = upper_semg.to(device,dtype=torch.float32)
            batch_y = hands_pose.to(device,dtype =torch.float32)
            pose_es = pose_es.float().to(device)
            # decoder input
            pred_len = net_config.pred_len - mask_len[0]
            label_len = net_config.label_len + mask_len[0]
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
            outputs,pose_es_pred = net(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if net_config.features == 'MS' else 0
            outputs = outputs[:, -pred_len:, f_dim:]
            batch_y = batch_y[:, -pred_len:, f_dim:].to(device)
            outputs = outputs.detach().numpy()
            batch_y = batch_y.detach().numpy()
            #每10帧画一次
            for frame in range(0,pred_len,10):
                pred_pose = outputs[0,frame,:]
                true_pose = batch_y[0,frame,:]
                name='error_'+str(i)+"_"+str(frame)+'.svg'
                draw_compare(pred_pose,true_pose,net_config.clim,name)
### 用于连续评估
def get_contine_result():
    device = torch.device("cpu")
    net = DoubleDecoder.Model(net_config).to(device)
    net.load_state_dict(torch.load(net_config.model_path))
    net.eval()
    all_list = net_config.shuffle_list
    device = torch.device("cpu")
    file_path = net_config.data_path
    scaler = MinMaxScaler()
    rms_loss = [];mae_loss = [];delta_median_loss = [];delta_mean_loss = [];delta_rr_loss = [];delta_cc_loss = []
    #读取一个人的数据做连续评估
    for i in range(1,11,3):
        upper_semgs,lowwer_semgs,accs,hands_poses = torch.load(os.path.join(file_path,"subject{0}.pt").format(i)).values()
        upper_semgs = torch.stack(upper_semgs)
        hands_poses = torch.stack(hands_poses)
        all_upper_semg_2d = upper_semgs.reshape(-1,upper_semgs.shape[-1])
        scaler.fit(all_upper_semg_2d)
        all_upper_semg_2d = scaler.transform(all_upper_semg_2d).reshape(upper_semgs.shape[0],upper_semgs.shape[1],upper_semgs.shape[2])
        all_hands_pose_2d = hands_poses
        mask_len = random.randint(0,5)

        seq_len = net_config.seq_len
        pred_len = net_config.pred_len - mask_len
        label_len = net_config.label_len + mask_len
        start_index = 0
        end_index = start_index+seq_len
        # rms_loss_exercise = [];mae_loss_exercise = [];delta_median_loss_exercise = [];delta_mean_loss_exercise = [];delta_rr_loss_exercise = [];delta_cc_loss_exercise = []
        for exercise in range(0,all_upper_semg_2d.shape[0],3):
            all_upper_semg = torch.tensor(all_upper_semg_2d[exercise,:,:]).unsqueeze(0)
            all_hands_pose = torch.tensor(all_hands_pose_2d[exercise,:,:]).unsqueeze(0)
            start_index = 0
            end_index = start_index+seq_len
            rms_loss_exercise = [];mae_loss_exercise = [];delta_median_loss_exercise = [];delta_mean_loss_exercise = [];delta_rr_loss_exercise = [];delta_cc_loss_exercise = []
            mask_len = 4
            while end_index <= all_upper_semg.shape[1]:
                upper_semg = all_upper_semg[:,start_index:end_index,:]
                hands_pose = all_hands_pose[:,start_index:end_index,:]
                batch_x_mark = None
                batch_y_mark = None 
                batch_x  = upper_semg.to(device,dtype=torch.float32)
                batch_y = hands_pose.to(device,dtype =torch.float32)
                pred_len = net_config.pred_len - mask_len
                label_len = net_config.label_len + mask_len
                if start_index > 0:
                    dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                    dec_inp = torch.cat([outputs[:, -label_len:, :], dec_inp], dim=1).float().to(device)
                else:
                    dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
                outputs,pose_es_pred = net(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if net_config.features == 'MS' else 0
                outputs = outputs[:, -pred_len:, f_dim:]
                batch_y = batch_y[:, -pred_len:, f_dim:].to(device)
                #计算评价参数
                #输出值与真实值的对比 采用 rms，mae,中值，均值，r2，CC
                rms = torch.sqrt(torch.mean(torch.pow(outputs - batch_y, 2)))
                mae = torch.mean(torch.abs(outputs - batch_y))
                outputs_median = torch.median(outputs)
                batch_y_median = torch.median(batch_y)
                delta_median = torch.abs(outputs_median - batch_y_median)
                outputs_mean = torch.mean(outputs)
                batch_y_mean = torch.mean(batch_y)
                delta_mean = torch.abs(outputs_mean - batch_y_mean)
                r2 = 1 - torch.sum(torch.pow(outputs - batch_y, 2)) / torch.sum(torch.pow(batch_y - torch.mean(batch_y), 2))
                x = outputs.flatten()
                y = batch_y.flatten()
                cc = (torch.sum((x - torch.mean(x)) * (y - torch.mean(y))) / torch.sqrt(torch.sum((x - torch.mean(x)) ** 2) * torch.sum((y - torch.mean(y)) ** 2)))
                print("exersice: %d, rms: %.4f, mae: %.4f, delta_median: %.4f, delta_mean: %.4f, r2: %.4f, cc: %.4f" % (exercise,rms.item(),mae.item(),delta_median.item(),delta_mean.item(),r2.item(),cc.item()))
                rms_loss.append(rms.item());mae_loss.append(mae.item());delta_median_loss.append(delta_median.item());delta_mean_loss.append(delta_mean.item());delta_rr_loss.append(r2.item());delta_cc_loss.append(cc.item())
                rms_loss_exercise.append(rms.item());mae_loss_exercise.append(mae.item());delta_median_loss_exercise.append(delta_median.item());delta_mean_loss_exercise.append(delta_mean.item());delta_rr_loss_exercise.append(r2.item());delta_cc_loss_exercise.append(cc.item())
                # 更新
                mask_len = 4
                start_index = end_index - mask_len
                end_index = start_index + seq_len
                if end_index > all_upper_semg.shape[1]:
                    # rms_loss.append(rms.item());mae_loss.append(mae.item());delta_median_loss.append(delta_median.item());delta_mean_loss.append(delta_mean.item());delta_rr_loss.append(r2.item());delta_cc_loss.append(cc.item())
                    break
            _rms = np.mean(rms_loss_exercise);_mae = np.mean(mae_loss_exercise);_delta_median = np.mean(delta_median_loss_exercise);_delta_mean = np.mean(delta_mean_loss_exercise);_delta_rr = np.mean(delta_rr_loss_exercise);_delta_cc = np.mean(delta_cc_loss_exercise)
            print('exercise: %.1d ,RMS: %.4f, MAE: %.4f, delta_median: %.4f, delta_mean: %.4f, delta_rr: %.4f, delta_cc: %.4f' % (exercise,_rms,_mae,_delta_median,_delta_mean,_delta_rr,_delta_cc))
    _rms = np.mean(rms_loss);_mae = np.mean(mae_loss);_delta_median = np.mean(delta_median_loss);_delta_mean = np.mean(delta_mean_loss);_delta_rr = np.mean(delta_rr_loss);_delta_cc = np.mean(delta_cc_loss)
    print('RMS: %.4f, MAE: %.4f, delta_median: %.4f, delta_mean: %.4f, delta_rr: %.4f, delta_cc: %.4f' % (_rms,_mae,_delta_median,_delta_mean,_delta_rr,_delta_cc))
    result = np.array([_rms,_mae,_delta_median,_delta_mean,_delta_rr,_delta_cc])
    np.save("./result/result_continue"+".npy",result)
                
            
class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
        
    def forward(self, p, q):
        p = F.log_softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)
        # print(p[0,0,:])
        # print(q[0,0,:])
        loss = F.kl_div(p,q,reduction='batchmean')
        return loss  



if __name__ == '__main__':
    # 计算误差
    # get_test_result()
    # 绘制误差图像
    draw_error_picture()
    # 连续评估
    # get_contine_result()





