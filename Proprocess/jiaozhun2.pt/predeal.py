subject = 'subject'
num = list([8,9,10])
angle ='_angle'
import openpyxl
import pandas as pd
import numpy as np
import datetime
import torch 
for i in num:
    subjects = 'subject_' + str(i) +'.xlsx'
    df = pd.read_excel(subjects)    
    new_df = df[['AGARRE','chan1','chan2','chan3','chan4','chan5','chan6','chan7','chan8','acc_x','acc_y','acc_z','R_CMC1_F','R_CMC1_A','R_MCP1_F','R_MCP2_F','R_MCP3_F','R_MCP4_F','R_MCP4_A','R_MCP5_F','R_MCP5_A','R_IP1_F','R_PIP2_F','R_PIP3_F','R_PIP4_F','R_PIP5_F','R_DIP2','R_DIP3','R_DIP4','R_DIP5','R_WR_F','R_WR_A']]
    array_df = np.array(new_df)
    # array_df = np.hstack([np.expand_dims(np.array(timestamp),1),array_df])
    # array_dfs = np.concatenate((np.expand_dims(np.array(timestamp),1),array_df),0)
    # new_subjects = 'subject' + str(i) + '_new' +'.npy'
    # np.save(new_subjects,array_dfs[1:])
    zero_row_idx =np.where(array_df[1:,0] == 0)[0]
    array_dfs = np.delete(array_df,zero_row_idx,axis=0)
    array_dfs = array_dfs[1:,1:]
    new_tensor = torch.from_numpy(array_dfs.copy())
    torch.save(new_tensor,'subject_' + str(i) +'.pt')

    print("{times} 被完成了".format(times = i))
