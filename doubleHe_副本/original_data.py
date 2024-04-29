import numpy as np
import matplotlib.pyplot as plt

# ORIGINAL_DATA_PATH  = './data/'
# TIMEDT_PATH         = ORIGINAL_DATA_PATH + 'timedt.data.{Temperature}'
# G_PATH              = ORIGINAL_DATA_PATH + 'G.{Temperature}'
# ADDHE_PATH          = ORIGINAL_DATA_PATH + 'Dump/dump.addHe.{Temperature}'
# MIGRATION_PATH      = ORIGINAL_DATA_PATH + 'MigrationLabel/migration_{File}.{Temperature}'


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
            mig_label.append(bool(int(line.strip())))
        mig_label = np.array(mig_label)
        
    # with open(MIGRATION_PATH.format(File='index', Temperature=temperature), 'r', encoding='utf-8') as fin:
    #     mig_index = []
    #     for i, line in enumerate(fin.readlines()[1:]):
    #         mig_index.append(int(line.strip()))
    #     mig_index = np.array(mig_index)
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

    t = np.array(t)
    msd = np.array(msd)
    csp = np.array(csp)
    xyz = np.array(xyz)
    r_ = np.sqrt(np.array(r_))
    v_xyz = np.array(v_xyz)
    v_ = np.sqrt(np.array(v_))
    angle = np.arccos(v_xyz / v_.reshape(len(t), 1))
    g = np.array(g)

    return t, msd, csp, xyz, r_, v_xyz, v_, angle, g

if __name__ == '__main__':
    pass
