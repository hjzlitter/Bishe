# from data import get_original_data, PosFeature, NegFeature
import numpy as np

import numpy as np
import pandas as pd
import sys
# import seaborn as sns
from numpy import random

# from matplotlib import pyplot as plt
import sys
import time
import math
from sklearn.preprocessing import StandardScaler
import optuna
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader

from ovito.io import *
from ovito.data import *
from ovito.modifiers import ExpressionSelectionModifier, DeleteSelectedModifier, \
InvertSelectionModifier, ExpandSelectionModifier
#, CoordinationAnalysisModifier, WignerSeitzAnalysisModifier, ClusterAnalysisModifier, ConstructSurfaceModifier,
from ovito.pipeline import *


import glob
import numpy as np
# import matplotlib.pyplot as plt

ORIGINAL_DATA_PATH  = './data/'
TIMEDT_PATH         = ORIGINAL_DATA_PATH + 'temper.{Temperature}/timedt.data.{Temperature}'
G_PATH              = ORIGINAL_DATA_PATH + 'G.{Temperature}'
MIGRATION_PATH      = ORIGINAL_DATA_PATH + 'MigrationLabel/migration_{File}.{Temperature}'



def NearestModify(Pipeline):
    Pipeline.modifiers.append(ExpressionSelectionModifier(expression='ParticleType == 2'))
    Pipeline.modifiers.append(ExpandSelectionModifier(mode=ExpandSelectionModifier.ExpansionMode.Nearest, num_neighbors=4))
    Pipeline.modifiers.append(InvertSelectionModifier())
    Pipeline.modifiers.append(DeleteSelectedModifier())
    return
    
def Nearest4PID(InputFile, ):
    pipeline = import_file(InputFile)
    NearestModify(pipeline)
    Pid_List = []
    Pos_List = []
    for nframe in range(pipeline.source.num_frames):
        Data = pipeline.compute(nframe)
        Ptype = Data.particles["Particle Type"]
        Pid = Data.particles['Particle Identifier']
        Pos = Data.particles['Position']
        mean = np.mean(Pos, axis=0, keepdims=True) 
        std = np.std(Pos, axis=0, keepdims=True)
        Pos = (Pos - mean) / std 
        Pid_tmp = list(Pid[Ptype == 1])
        Pid_List.append(Pid_tmp)
        Pos_List.append(Pos[Ptype == 1])
    return Pos_List, Pid_List

def PosFeature(Temper, FilePath, n):
    Pid_List = []
    Pos_list = []
    FileNum = len(glob.glob(f"{FilePath}/dump_temp*"))
    Time_List = list(range(FileNum*1000+1))
    for i in range(1, FileNum+1):
        Input_File = f"{FilePath}/dump_temp.{Temper}.{i}"  
        # print(type(Input_File))
        Pos_tmp, Pid_tmp = Nearest4PID(Input_File)
        if i != 1:
            del(Pid_tmp[0])
            del(Pos_tmp[0])
        Pid_List += Pid_tmp
        Pos_list += Pos_tmp
    # print(type(Pid_tmp))
    Pos_Feature = []
    Time_Feature = []
    for t in range(1, len(Pid_List)-1):
        InterSection = list(set(Pid_List[t-1]) & set(Pid_List[t]))
        if len(InterSection) == 4:
            continue
        else:
            if(t+n<len(Pid_List)-1):
                if set(Pid_List[t+n]) == set(Pid_List[t-1]):
                    continue
                else:
                    pos = np.stack([Pos_list[t-i] for i in range(1,6)],axis=0)
                    Pos_Feature.append(pos)
                    Time_Feature.append(t) 
    Pos_feature = np.stack(Pos_Feature, axis=0)  
    return Pos_feature, Time_Feature

def NegFeature(Temper, FilePath, n, num_need):
    Pid_List = []
    Pos_list = []
    FileNum = len(glob.glob(f"{FilePath}/dump_temp*"))
    Time_List = list(range(FileNum*1000+1))
    for i in range(1, FileNum+1):
        Input_File = f"{FilePath}/dump_temp.{Temper}.{i}"  
        Pos_tmp, Pid_tmp = Nearest4PID(Input_File)
        if i != 1:
            del(Pid_tmp[0])
            del(Pos_tmp[0])
        Pid_List += Pid_tmp
        Pos_list += Pos_tmp
    Pos_Feature = []
    Time_Feature = []
    while len(Pos_Feature) < num_need:
        rand_num =  np.random.randint(1, len(Pid_List)-1)
        InterSection = list(set(Pid_List[rand_num-1]) & set(Pid_List[rand_num]))
        if len(InterSection) == 4:
            Pos_Feature.append([Pos_list[rand_num-i] for i in range(1,6)])
            Time_Feature.append(rand_num)
        else:
            continue 
    Pos_feature = np.stack(Pos_Feature, axis=0)
    return Pos_feature, Time_Feature

# def get_migration_label(
#     temperature:int, # 温度
# ):
#     """
#     input: 
#         temperature 

#     return: 
#         mig_label 
#     """
#     with open(MIGRATION_PATH.format(File='label', Temperature=temperature), 'r', encoding='utf-8') as fin:
#         mig_label = []
#         for i, line in enumerate(fin.readlines()[1:]):
#             mig_label.append(bool(int(line.strip())))
#         mig_label = np.array(mig_label)

#     return mig_label


def get_original_data(
    temperature:int,# 温度
    time_index # 时间序列
):
    """
    input temperature : 温度
    
    return t, msd, csp, xyz, r_, v_xyz, v_, angle, g

    t       时间序列(单位:ps)
    msd     单He的msd(均方位移)
    csp     CSP(中心对称参数)
    xyz     单He的xyz坐标
    r_      单He离原点距离
    v_xyz   单He的沿xyz坐标的速度分量
    v_      单He的速度大小
    """
    with open(TIMEDT_PATH.format(Temperature=temperature), 'r', encoding='utf-8') as fin:
        t = []      # 时间序列(单位:ps)
        msd = []    # 单He的msd(均方位移)
        csp = []    # CSP(中心对称参数)
        xyz = []    # 单He的xyz坐标
        r_ = []     # 单He离原点距离
        v_xyz = []  # 单He的沿xyz坐标的速度分量
        v_ = []     # 单He的速度大小
        for i, line in enumerate(fin.readlines()[1:]):
            data = list(map(float, line.strip().split(' ')))
            t.append(data[0])
            msd.append(data[1])
            csp.append(data[2:8])
            xyz.append(data[8:11])
            r_.append(data[11])
            v_xyz.append(data[12:15])
            v_.append(data[15])

    # with open(G_PATH.format(Temperature=temperature), 'r', encoding='utf-8') as fin:
    #     g = []      # g参数
    #     for i, line in enumerate(fin.readlines()[1:]):
    #         data = list(map(float, line.strip().split(' ')))
    #         g.append(data[1:7])

    t = np.array(t)
    msd = np.array(msd)
    csp = np.array(csp)
    xyz = np.array(xyz)
    r_ = np.sqrt(np.array(r_))
    v_xyz = np.array(v_xyz)
    v_ = np.sqrt(np.array(v_))
    angle = np.arccos(v_xyz / v_.reshape(len(t), 1))
    # g = np.array(g)

    csp_re = [np.stack(csp[index-5:index],axis=0) for index in time_index]
    xyz_re = [np.stack(xyz[index-5:index],axis=0) for index in time_index]
    r_re = [np.stack(r_[index-5:index],axis=0) for index in time_index]
    v_xyz_re = [np.stack(v_xyz[index-5:index],axis=0) for index in time_index]
    v_re = [np.stack(v_[index-5:index],axis=0) for index in time_index]
    angle_re = [np.stack(angle[index-5:index],axis=0) for index in time_index]
    # g_re = [g[index-5:index] for index in time_index]
    csp_re = np.stack(csp_re, axis=0)
    xyz_re = np.stack(xyz_re, axis=0)
    r_re = np.stack(r_re, axis=0)
    r_re = np.expand_dims(r_re, axis=-1)
    v_xyz_re = np.stack(v_xyz_re, axis=0)
    v_re = np.stack(v_re, axis=0)
    v_re = np.expand_dims(v_re, axis=-1)
    angle_re = np.stack(angle_re, axis=0)
    # g_re = np.stack(g_re, axis=0)
    # return csp_re, xyz_re, r_re, v_xyz_re, v_re, angle_re#, g_re
    return csp_re, xyz_re, r_re, v_xyz_re, v_re, angle_re


# ITER_PATH = './Datasets/{train_or_test}_iter_{temperature}.pth'
PATH = './output/'
LOSS_PATH = PATH + 'loss/loss_{temperature}.txt'
RESULT_PATH = PATH + 'result/result_{temperature}.txt'
MODEL_PATH = PATH + 'model/{model}_{temperature}_{epoch}.pkl'
LOG_PATH = PATH + 'log/{date}_{information}.txt'
LOG_FIGURE_PATH = PATH + 'log/{date}_{information}.png'
DATA_PATH =  './data/'
TEMPERATURE = 400
LEARNING_RATE = 6e-3
# TRY_TIMES = 10
# TEMPERATURE = 150
# MAX = 1.0
# MIN = 0.5
# TIME_LENGTH = 50
STRIDE = 5
STEADY_LENGTH = 100
MIGRATION_LENGTH = 50
EMBEDDING_SIZE = 12
N_TRIAL = 100
INPUT_SIZE = 15
BATCH_SIZE = 16
EPOCHS = 1000

# DEVICE = torch.device('mps' if torch.cuda.is_available() else 'cpu')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_train_iter(
    Temperature, 
    Batch_size, 
    path
):
    # print(index)
    # print(data_num)
    
    pos_data, time_data = PosFeature(Temperature, f"{path}/temper.{Temperature}", 30)
    data = get_original_data(Temperature, time_data)
    # print(pos_data.shape)
    pos_data = pos_data.reshape(pos_data.shape[0], pos_data.shape[1], -1)
    data = np.concatenate(data, axis=2)
    # print(data.shape)
    # print(pos_data.shape)
    pas_data = np.concatenate([data, pos_data], axis=2)
    
    neg_pos_data, neg_time_data = NegFeature(Temperature, f"{path}/temper.{Temperature}", 30, len(time_data))
    neg_data = get_original_data(Temperature, neg_time_data)
    # print(neg_pos_data.shape)
    neg_pos_data = neg_pos_data.reshape(neg_pos_data.shape[0], neg_pos_data.shape[1], -1)
    
    neg_data = np.concatenate(neg_data, axis=2)
    neg_data = np.concatenate([neg_data, neg_pos_data], axis=2)
    
    pas_y  = np.ones((pas_data.shape[0], 1))
    neg_y  = np.zeros((neg_data.shape[0], 1))

    X = np.concatenate([pas_data, neg_data], axis=0)
    Y = np.concatenate([pas_y, neg_y], axis=0)


    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    X = X.unsqueeze(1)

    data_num = X.shape[0]
    shuffled_index = np.random.permutation(range(data_num))
    X = X[shuffled_index]
    Y = Y[shuffled_index]
    Train_X, Test_X = X[:int(data_num*0.8)], X[int(data_num*0.8):]
    Train_Y, Test_Y = Y[:int(data_num*0.8)], Y[int(data_num*0.8):]

    Train_generator = DataLoader(
        torch.utils.data.TensorDataset(Train_X, Train_Y), 
        Batch_size, 
        shuffle=True
    )
    Test_generator = DataLoader(
        torch.utils.data.TensorDataset(Test_X, Test_Y), 
        Batch_size, 
        shuffle=True
    )
    return Train_generator, Test_generator


class CNN(nn.Module):
    def __init__(
        self, 
        Channel_in, 
        Height_in, 
        Width_in, 
        Output_size, 
        Filter_num, 
        Kernel_list, 
        dropout = 0.5, 
    ):
        super(CNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    Channel_in, 
                    Filter_num, 
                    kernel_size=(kernel, Width_in), 
                    padding=((kernel - 1) // 2, 0), 
                ),
                nn.ReLU(),
                nn.MaxPool2d(
                    kernel_size=((Height_in+3)//4, 1), 
                    stride=(Height_in+3)//4, 
                    padding=((Height_in-Height_in//4*4+1)//2, 0), 
                ), 
            )
            for kernel in Kernel_list
        ])
        self.fc = nn.Sequential(
            nn.Dropout(dropout), 
            nn.Linear(Filter_num * len(Kernel_list) * 3, 64),
            nn.ReLU(), 
            nn.Linear(64, 16), 
            nn.Linear(16, Output_size)
        )

    def forward(self, x):
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out



def train(
    net:nn.Module, 
    Train_generator, 
    loss_func,
    optimizer, 
    scheduler, 
    device
):
    net = net.to(device)
    net.train()
    sum_loss = []
    
    Train_generator =  DataLoader(Train_generator.dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)
    for x, y in Train_generator:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        scores = net(x)
        loss = loss_func(scores, y)
        sum_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    return np.array(sum_loss).mean()
    
def evaluate(
    net:nn.Module, 
    Test_generator, 
    loss_func,
    optimizer, 
    scheduler, 
    device
):
    sum_loss = []
    net.eval()
    Test_generator = DataLoader(Test_generator.dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)
    with torch.no_grad():
        for x, y in Test_generator:
            x = x.to(device)
            y = y.to(device)
            scores = net(x)
            loss = loss_func(scores, y)
            sum_loss.append(loss.item())
        
    return np.array(sum_loss).mean()


def train_multi_epochs(
    net:nn.Module, 
    Train_generator, 
    Test_generator,     
    loss_func,
    optimizer, 
    scheduler, 
    epochs, 
    device, 
    information:str, 
    show_train_process=None
):
    best_epoch = 0
    best_test_loss = 9.9e9
    best_net = net.state_dict()
    sum_train_loss, sum_test_loss = [], []
    date0 = time.strftime('%Y-%m-%d %a %H-%M-%S', time.localtime(time.time()))

    with open(LOG_PATH.format(date=date0, information=information), 'w') as log_fin:
        log_fin.write(information + '\n')
        log_fin.write('epoch' + ' ' + 'train_loss' + ' ' + 'test_loss' + ' ' + 'time' + ' ' + 'best_epoch' + '\n')
        t1 = time.time()
        for epoch in range(epochs):
            t0 = time.time()
            train_loss = train(net, Train_generator, loss_func, optimizer, scheduler, device).item()
            test_loss = evaluate(net, Test_generator, loss_func, optimizer, scheduler, device).item()

            sum_train_loss.append(train_loss)
            sum_test_loss.append(test_loss)

            if epoch == 0 or test_loss < best_test_loss:
                best_epoch = epoch
                best_test_loss = test_loss
                best_net = net.state_dict()

            log_fin.write(str(epoch) + ' ' + str(train_loss) + ' ' + str(test_loss) + ' ' + str(time.time()-t0) + ' ' + str(best_epoch) + '\n')
            if show_train_process != None and epoch % show_train_process == 0:
                print('epoch={:>4}, train_loss= {:.4f}, test_loss= {:.4f}, time= {:.2f}sec, best_epoch= {:>4}'.format(epoch, train_loss, test_loss, time.time()-t1, best_epoch))
                t1 = time.time()
        
        log_fin.write('\n')
        log_fin.write('best_epoch=' + str(best_epoch) + '\n')
        log_fin.write('best_test_loss=' + str(best_test_loss) + '\n')
    return best_test_loss, best_epoch, best_net, sum_train_loss, sum_test_loss




def singal_train(lr, ga, dropout=0.5):

    train_iter, test_iter = get_train_iter(TEMPERATURE, BATCH_SIZE, DATA_PATH)

    criteon = nn.BCEWithLogitsLoss().to(DEVICE)
    net = CNN(1, 5, 29, 1, 32, [9, 7, 5, 3], dropout)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1.0)
    scheduler = MultiStepLR(optimizer, [1000, 3000], gamma=ga, last_epoch=-1)
    
    best_test_loss, best_epoch, _, _, _ = train_multi_epochs(net, train_iter, test_iter, criteon, optimizer, scheduler, EPOCHS, DEVICE, 'lr = {:.8f}, gama = {:.4f}, dropout = {:4f}'.format(lr, ga, dropout), show_train_process=1)
    
    return best_test_loss


if __name__ == "__main__":
    singal_train(LEARNING_RATE, 0.5, 0.5)