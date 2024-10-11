import torch
import random
import numpy as np
class net_config:
    #  输入输出参数
    all_channels = False
    if all_channels == False:
        enc_in = 8  # 原来应该是8 如果采用泛化效果 省略掉模型
        dec_in = 20  # decoder 为上一个时间点的输出
        es_out = 20
        c_out = 20
    else:
        enc_in = 16 # 原来应该是16 如果更改采用泛化效果 省略掉模型
        dec_in =20 #decoder 为上一个时间点的输出
        es_out = 20
        c_out = 20
    # data_path = "dataset/myo"
    # checkpoint_path ="checkpoint/"
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = ("cpu")
    ## 数据噪声
    std =  0.04
    mean = 0

    # 模型参数
    d_model = 512
    rnn_hidden = 128
    n_heads = 8
    e_layers = 2
    d_layers = 1
    d_ff =2048
    moving_avg = 25 #window size of moving average
    factor = 3 # attn factor
    distil = True ###压缩模型 蒸馏模型
    dropout = 0.2
    dropout_linear = 0.4
    dropout_rnn = 0.2
    embed = "fixed"
    activation  = 'gelu'
    output_attention =False
    d0_predict = False
    features = 'M'
    freq = 'ms'
    reduction = 4
    assert d_model // (reduction**2) >  0 ,"reduction is too large"
    #优化参数
    patience = 5
    batch_size = 128
    epochs =300
    learning_rate = 0.0001
    fix_seed = 2024
    lradj = 'type1'
    use_amp = False

    #显卡设置
    usegpu = True
    gpu = 0
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # 数据集

    # itransfomer
    size = [100,1,99]
    random_mask = size[0]/20
    # size = [40,1,39]
    seq_len = size[0];label_len = size[1];pred_len = size[2]
    exp_name = 'MTSF'
    channel_independence = False
    inverse = False
    class_strategy ='projection'
    data_path = "dataset/myo"
    checkpoint_path ="checkpoint/"
    # model_path = "checkpoint/0430/epoch_141.pth"
    visual_path ="visual/"
    efficient_training = False
    use_norm = True
    partial_train =False
    partial_start_index =0
    signaldata_len = 1000
    shuffle_list = [i for i in range(1,11)]
    random.shuffle(shuffle_list)
    train_list = shuffle_list[:7]
    train_extra = shuffle_list[6:7]
    valid_list = shuffle_list[7:8]
    test_list = shuffle_list[8:]
    np.save('shuffle_list.npy',shuffle_list)
    outfile = "result\exercise_test.mp4"

    # 损失权重
    w1 = 1
    w2 = 1

    clim = [-0.05,0.05]
    model_path = "checkpoint\epoch_41.pth"
