# import
import glob
import numpy as np


import numpy as np
import pandas as pd
import sys
# import seaborn as sns
from numpy import random

from matplotlib import pyplot as plt
import sys
import time
import math
from sklearn.preprocessing import StandardScaler
# import original_data
import optuna
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset

# config
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
EPOCHS = 100
N_FRAME = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data
def get_migration_label(
    path:str
):
    """
    input: 
        temperature 

    return: 
        mig_label 
    """
    with open(path, 'r', encoding='utf-8') as fin:
        mig_label = []
        for i, line in enumerate(fin.readlines()[1:]):
            parts = line.strip().split()
            mig_label.append(bool(int(parts[1])))
        mig_label = np.array(mig_label)
    return mig_label

def get_original_data(
   path:str
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
    with open(path, 'r', encoding='utf-8') as fin:
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
    indices_to_remove = np.arange(1001, len(t) - 1, 1001)
    t = np.array(t)
    t = np.delete(t, indices_to_remove)
    t = t.reshape(-1, 1)
    msd = np.array(msd)
    msd = np.delete(msd, indices_to_remove)
    msd = msd.reshape(-1, 1)
    csp = np.array(csp)
    csp = np.delete(csp, indices_to_remove, axis=0)
    xyz = np.array(xyz)
    xyz = np.delete(xyz, indices_to_remove, axis=0)
    r_ = np.sqrt(np.array(r_))
    r_ = np.delete(r_, indices_to_remove)
    r_ = r_.reshape(-1, 1)
    v_xyz = np.array(v_xyz)
    v_xyz = np.delete(v_xyz, indices_to_remove, axis=0)
    v_ = np.sqrt(np.array(v_))
    v_ = np.delete(v_, indices_to_remove)
    v_ = v_.reshape(-1, 1)
    angle = np.arccos(v_xyz / v_.reshape(len(t), 1))
    # g = np.array(g)

    return t, msd, csp, xyz, r_, v_xyz, v_, angle# , g

data1 = get_original_data('./temper.300/timedt.dataHe1.300')
data2 = get_original_data('./temper.300/timedt.dataHe2.300')
data1 = np.hstack(data1)
data2 = np.hstack(data2)
data = np.hstack((data1,data2[:,1:]))

label1 = get_migration_label('./y_label_2001.txt').reshape(-1, 1)
label2 = get_migration_label('./y_label_2002.txt').reshape(-1, 1)
labels = np.column_stack((label1, label2))

positive10_indices = np.where((labels[:, 0] == 1) & (labels[:, 1] == 0))[0]
positive01_indices = np.where((labels[:, 0] == 0) & (labels[:, 1] == 1))[0]
positive11_indices = np.where((labels[:, 0] == 1) & (labels[:, 1] == 1))[0]
negative_indices = np.where((labels[:, 0] == 0) & (labels[:,1]==0))[0]
n_samples = min(len(positive10_indices), len(positive01_indices), len(positive11_indices), len(negative_indices))
np.random.shuffle(positive10_indices)
np.random.shuffle(positive01_indices)
np.random.shuffle(positive11_indices)
np.random.shuffle(negative_indices)

selected_positive01_indices = positive01_indices[:n_samples]
selected_positive10_indices = positive10_indices[:n_samples]
selected_positive11_indices = positive11_indices[:n_samples]
selected_negative_indices = negative_indices[:n_samples]

selected_indices = np.sort(np.concatenate([selected_positive01_indices, selected_positive10_indices, selected_positive11_indices, selected_negative_indices]))




final_data = []
for idx in selected_indices:
    start_idx = max(idx - N_FRAME + 1, 0)  
    extracted_data = data[start_idx:start_idx + N_FRAME] 
    final_data.append(extracted_data)
final_data = np.array(final_data)

final_labels = []
for idx in selected_indices:
    start_idx = max(idx - N_FRAME + 1, 0)  
    extracted_data = labels[start_idx + N_FRAME] 
    final_labels.append(extracted_data)
final_labels = np.array(final_labels)

final_data = final_data.reshape(-1, 1, N_FRAME, 38)
final_data = torch.tensor(final_data, dtype=torch.float32)
final_labels = torch.tensor(final_labels, dtype=torch.float32)


# MODEL
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
        # print(Kernel_list)
        self.fc = nn.Sequential(
            nn.Dropout(dropout), 
            nn.Linear(Filter_num * len(Kernel_list) * 4, 64),
            nn.ReLU(), 
            nn.Linear(64, 16), 
            nn.Linear(16, Output_size)
        )
        self.output = nn.Sigmoid()

    def forward(self, x):
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(x.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        output = self.output(out)
        return out

# TRAIN
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
    for x, y in Test_generator:
        x = x.to(device)
        y = y.to(device)
        scores = net(x)
        loss = loss_func(scores, y)
        sum_loss.append(loss.item())
    
    return np.array(sum_loss).mean()


def train_multi_epochs(
    path, 
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

    with open(path.format(date=date0, information=information), 'w') as log_fin:
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


def singal_train_CNN(lr=LEARNING_RATE, ga=0.5, dropout=0.5):
    t1 = time.time()
    # train_iter, test_iter = get_train_iter(TEMPERATURE, BATCH_SIZE)
    train_dataset = TensorDataset(final_data, final_labels)
    train_iter = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)
    test_dataset = TensorDataset(final_data, final_labels)
    test_iter = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)
    # test_iter = DataLoader([final_data, final_labels], batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)
    criteon = nn.BCEWithLogitsLoss().to(DEVICE)
    # net = CNN(1, TIME_LENGTH, INPUT_SIZE, 1, 32, [3, 5, 7, 9], dropout)
    net = CNN(1,N_FRAME, 37,2, 8, [9,7,5,3], dropout)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=0)
    scheduler = MultiStepLR(optimizer, [int(EPOCHS*0.2), int(EPOCHS*0.4)], ga, last_epoch=-1)

    best_test_loss, best_epoch, _, multi_train_loss, multi_test_loss = train_multi_epochs('./output/1', net, train_iter, test_iter, criteon, optimizer, scheduler, EPOCHS, DEVICE, 'Temperature={}'.format(300), show_train_process=10)

    print('best_test_loss= {:.4f}, best_epoch= {:>4}, time= {:.2f}sec'.format(best_test_loss, best_epoch, time.time()-t1))

    plt.plot(multi_train_loss)
    plt.plot(multi_test_loss)
    plt.show()
    
    return best_test_loss

if __name__ == '__main__':
    singal_train_CNN()