from ovito.io import *
from ovito.data import *
from ovito.modifiers import ExpressionSelectionModifier, DeleteSelectedModifier, \
InvertSelectionModifier, ExpandSelectionModifier
#, CoordinationAnalysisModifier, WignerSeitzAnalysisModifier, ClusterAnalysisModifier, ConstructSurfaceModifier,
from ovito.pipeline import *


import glob
import numpy as np
# import matplotlib.pyplot as plt



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


if __name__ == "__main__":
    # print(index)
    # print(data_num)
    Temperature = 400
    path = '.'
    
    # 下面有个30帧的参数，可以调整
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
    file_path = 'data.npz'
    np.savez(file_path, X=X, Y=Y)
    
    
    

    


    # X = torch.tensor(X, dtype=torch.float32)
    # Y = torch.tensor(Y, dtype=torch.float32)
    # X = X.unsqueeze(1)

    # data_num = X.shape[0]
    # shuffled_index = np.random.permutation(range(data_num))
    # X = X[shuffled_index]
    # Y = Y[shuffled_index]