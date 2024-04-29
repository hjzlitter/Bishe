from ovito.io import *
from ovito.data import *
from ovito.modifiers import ExpressionSelectionModifier, DeleteSelectedModifier, \
InvertSelectionModifier, ExpandSelectionModifier
#, CoordinationAnalysisModifier, WignerSeitzAnalysisModifier, ClusterAnalysisModifier, ConstructSurfaceModifier,
from ovito.pipeline import *
import glob
import numpy as np

def NearestModify(Pipeline):
    Pipeline.modifiers.append(ExpressionSelectionModifier(expression='ParticleType==2 && ParticleIdentifier==2001'))
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
        Input_File = f"{FilePath}/dump_temp.{Temper}.{i}"  
        Pid_tmp = Nearest4PID(Input_File)
        if i != 1:
            del(Pid_tmp[0])
        Pid_List += Pid_tmp
    Mig_Label = [0]

    for t in range(1, len(Pid_List)):
        if set(Pid_List[t-1])==set(Pid_List[t]):
            Mig_Label += [0]
        else:
            Mig_Label += [1]
    return Time_List, Mig_Label

if __name__ == "__main__":
    temper = 300 
    path = "."
    time_series, y_label = MigrationLabel(temper, f"{path}/temper.{temper}")
    save_data = np.column_stack((np.array(time_series), np.array(y_label)))
    np.savetxt(f"{path}/y_label_2001.txt", save_data, fmt='%d', header='time mig_label')