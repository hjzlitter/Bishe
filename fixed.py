from ovito.io import *
from ovito.data import *
from ovito.modifiers import ExpressionSelectionModifier, DeleteSelectedModifier, \
InvertSelectionModifier, ExpandSelectionModifier
#, CoordinationAnalysisModifier, WignerSeitzAnalysisModifier, ClusterAnalysisModifier, ConstructSurfaceModifier,
from ovito.pipeline import *
import glob
import numpy as np
import warnings
warnings.filterwarnings("ignore")

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
    for nframe in range(pipeline.source.num_frames):
        Data = pipeline.compute(nframe)
        Ptype = Data.particles["Particle Type"]
        Pid = Data.particles['Particle Identifier']
        Pid_tmp = list(Pid[Ptype == 1])
        Pid_List.append(Pid_tmp)
    return Pid_List


def MigrationLabel(Temper, FilePath, ):
    Pid_List = []
    FileNum = len(glob.glob(f"{FilePath}/dump_temp*"))
    Time_List = list(range(FileNum*1000+1))
    for i in range(1, FileNum+1):
        Input_File = f"{FilePath}/dump_temp.{Temper}.{i}"  # 输入文件
        Pid_tmp = Nearest4PID(Input_File)
        if i != 1:
            del(Pid_tmp[0])
        Pid_List += Pid_tmp
    Mig_Label = [0]
    """
    这里给出氦原子是否处于迁移路径上的label，
    Pid_List储存的元素为各个时刻氦原子所处四面体笼子的四个顶点原子的id，
    如果氦原子迁移则一定存在原子id变化，
    所以可以通过判断某一时刻前后两帧四原子id是否发生变化来判定这一时刻氦原子是否处于迁移路径上
    """
    for t in range(1, len(Pid_List)-1):
        InterSection = list(set(Pid_List[t-1]) & set(Pid_List[t+1]))
        if len(InterSection) == 4:
            Mig_Label += [0]
        else:
            Mig_Label += [1]
    Mig_Label += [0]
    return Time_List, Mig_Label



def MigrationLabel_fixed(Temper, FilePath, n):
    Pid_List = []
    FileNum = len(glob.glob(f"{FilePath}/dump_temp*"))
    Time_List = list(range(FileNum*1000+1))
    for i in range(1, FileNum+1):
        Input_File = f"{FilePath}/dump_temp.{Temper}.{i}"  # 输入文件
        Pid_tmp = Nearest4PID(Input_File)
        if i != 1:
            del(Pid_tmp[0])
        Pid_List += Pid_tmp
    Mig_Label = [0]
    """
    这里给出氦原子是否处于迁移路径上的label，
    Pid_List储存的元素为各个时刻氦原子所处四面体笼子的四个顶点原子的id，
    如果氦原子迁移则一定存在原子id变化，
    所以可以通过判断某一时刻前后两帧四原子id是否发生变化来判定这一时刻氦原子是否处于迁移路径上
    """
    for t in range(1, len(Pid_List)-1):
        InterSection = list(set(Pid_List[t-1]) & set(Pid_List[t+1]))
        if len(InterSection) == 4:
            # Check if the Pid_list will return to its original state after n steps
            Mig_Label += [0]
        else:
            if(t+n<len(Pid_List)-1):
                if set(Pid_List[t+n]) == set(Pid_List[t-1]):
                    Mig_Label += [0]
                else:
                    Mig_Label += [1]
    Mig_Label += [0]
    return Time_List, Mig_Label

if __name__ == "__main__":
    temper = 400 #温度
    path = "./data"
    n_delete = 40
    time_series, y_label = MigrationLabel(temper, f"{path}/temper.{temper}", n_delete)
    save_data = np.column_stack((np.array(time_series), np.array(y_label)))
    np.savetxt(f"{path}/y_label.txt", save_data, fmt='%d', header='time mig_label')