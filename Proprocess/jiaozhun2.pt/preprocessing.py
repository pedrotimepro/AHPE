import openpyxl
import pandas as pd
import numpy as np
import datetime
import torch 
"""
file_list: 数据名称
column_list: 数据列
data_len: 截取的数据长度,从后往前截取多少个数据点,200hz
"""
def preprocess_Db5(file_list,column_list,data_len =1100):
    for i in file_list:
        subjects = 'subject' + str(i) +'.xlsx'
        data = pd.read_excel(subjects)
        semg_datas = np.array(data[column_list])
        np.save("semg{}.npy".format(i),semg_datas)
def DB52pt(file_list,data_len =1000):
    for i in file_list:
        exercise = 0;temp = [];exercise_list = []
        high_arm = [];low_arm = [];acc = [];hands =[]
        subjects = 'semg{0}.npy'.format(i)
        datas = np.load(subjects)
        zero_row_idx =np.where(datas[1:,0] == 0)[0]
        del_row_idx = [i+1 for i in zero_row_idx]
        semg_data = np.delete(datas,del_row_idx,axis=0)
        new_zeros_row2 = np.where(semg_data[1:,0] == 0)[0]
        if new_zeros_row2.shape[0] == 0:
            semg_data = torch.from_numpy(semg_data.copy()).to(torch.float32)
            for row in range(0,semg_data.shape[0]):
                if semg_data[row,0] != exercise:
                    temp.append(row);exercise = semg_data[row,0]
            print(len(temp))
            for j in range(1,len(temp)-1):
                if temp[j]-temp[j-1] > data_len:
                    exercise_list.append(semg_data[temp[j]-data_len:temp[j],0])
                    high_arm.append(semg_data[temp[j]-data_len:temp[j],1:9])
            torch.save({'high_arm':high_arm,'exercise':exercise_list},'dataset/classical'+str(i)+'.pt')
            print(len(high_arm))
            print('Preprocessed subject{0} is saved at{1}'.format(i,'dataset/classical'+str(i)+'.pt'))

        # print(semg_data.shape[0])
        # for row in range(0,semg_data.shape[0]):
        #     if semg_data[row,0] != exercise:
        #         temp.append(row);exercise = semg_data[row,0]
        # high_arm = [];low_arm = [];hands =[]
        # for j in range(1,len(temp)-1):
        #     try:
        #     ### 不保存短的数据，且确定一定的数据长度
        #         if semg_data[temp[j]+2,0] != 0 and temp[j] - temp[j-1]>data_len:
        #             high_arm.append(semg_data[temp[j]-data_len:temp[j],1:9])
        #             low_arm.append(semg_data[temp[j]-data_len:temp[j],9:17])
        #             hands.append(semg_data[temp[j]-data_len:temp[j],17:])
        #     except:
        #         print("{0}行出现了问题".format(temp[j]))
        # torch.save({'high_arm':high_arm,'low_arm':low_arm,'hands':hands},'dataset/subject'+str(i)+'.pt')
        # print('Preprocessed subject{0} is saved at{1}'.format(i,'dataset/subject'+str(i)+'.pt'))
        






if __name__ == '__main__':

    file_list = [1,2,3,4,5,6,7,8,9,10]
    DB52pt(file_list=file_list)
    # column_lists = ['AGARRE','chan1','chan2','chan3','chan4','chan5','chan6','chan7','chan8'
    #                 ,'chan9','chan10','chan11','chan12','chan13','chan14','chan15','chan16',
    #                 'acc_x','acc_y','acc_z','R_CMC1_F','R_CMC1_A','R_MCP1_F','R_MCP2_F','R_MCP3_F',
    #                 'R_MCP4_F','R_MCP4_A','R_MCP5_F','R_MCP5_A','R_IP1_F',
    #                 'R_PIP2_F','R_PIP3_F','R_PIP4_F','R_PIP5_F','R_DIP2',
    #                 'R_DIP3','R_DIP4','R_DIP5','R_WR_F','R_WR_A']
    # preprocess_Db5(file_list,column_lists)
#    semg = np.load('semg1.npy')
#     print(semg.shape)
#     zero_row_idx =np.where(semg[1:,0] == 0)[0]
#     print(semg.shape[0]- len(zero_row_idx))
#     semg_data = np.delete(semg,zero_row_idx,axis=0)
#     print(semg_data.shape)
#     new_zeros_row = np.where(semg_data[1:,0] == 0)[0]
#     print(len(new_zeros_row)) 
    