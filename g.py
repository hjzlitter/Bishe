#Analysize the G1 G2 of atom He
#Specified to be called by shell command in lammps. Part 1
from ovito.io import *
from ovito.data import *
from ovito.modifiers import ExpressionSelectionModifier, DeleteSelectedModifier, ExpandSelectionModifier, \
    InvertSelectionModifier
from ovito.pipeline import *
import numpy as np
import math
import time
t0=time.time()
###################import the original dump file.
location="/data/liuliuli/g/singleHe_Bulk" #@@@@@@@@@@@@@location of dump file
temper = 400         #temperature
Rc=5.41           # The cutoff distance of Fe-He potential energy
etaa_g1=[1,0.05]     #list of η/pi/Rc**2 for g1
etaa_g2=[1,0.005]     #list of η/pi/Rc**2 for g2
lambdaa_g2=[1,-1] #list of λ for g2
kxi_g2=[1,2]      #list of ξ for g2
def cos(x,y): #The cos value of two R vectors.
    Lx=np.sqrt(x.dot(x))
    Ly=np.sqrt(y.dot(y))
    cos_value=x.dot(y)/(Lx*Ly)
    return cos_value
def Distance(x,y): #The distance between two R vectors, after the boundary transition.

    return (np.sum((x-y)**2))**0.5

def fcutg1(R0,Ri,etaa): #The cutoff function * g1
    #Rc=5.41 # The cutoff distance of Fe-He potential energy
    return math.exp(-etaa*math.pi*Distance(R0,Ri)**2/Rc**2)*(0.5*(math.cos(math.pi*Distance(R0,Ri)/Rc)+1))

def fcutg2(R0,Ri,Rj,etaa,kxi,lambdaa): #The cutoff function * g2
    #Rc=5.41 # The cutoff distance of Fe-He potential energy
    return 2**(1-kxi)*(1+lambdaa*cos(Ri-R0,Rj-R0))**kxi*fcutg1(Ri,Rj,etaa)*fcutg1(R0,Ri,etaa)*fcutg1(R0,Rj,etaa)

num = NUMBERREPLACED
lower = 100*(num-1) + 1
upper = 100*num + 1
for batch in range(lower,upper):#number of batches
    pipeline = import_file('{0}/temper.{1}/dump_temp.{1}.{2}'.format(location,temper,batch)) #@@@@@@@@@@@@@@@@@@@@@@inputfile
    pipeline.modifiers.append(ExpressionSelectionModifier(expression = 'ParticleType == 2'))
    pipeline.modifiers.append(ExpandSelectionModifier(cutoff = 5.41))
    pipeline.modifiers.append(InvertSelectionModifier())
    pipeline.modifiers.append(DeleteSelectedModifier())
    G=np.zeros((pipeline.source.num_frames,7)) # nstep, G1 (2), G2 (6)
    for nframe in range(pipeline.source.num_frames):#
        data = pipeline.compute(nframe)
        ptype=data.particles["Particle Type"]
        pid = data.particles["Particle Identifier"]
        pos_all = data.particles_["Position_"]
        #Exclude the boundary condition
        for index in range(3):
            if(np.max(pos_all, axis=0)[index]-np.min(pos_all, axis=0)[index]>data.cell[index,index]/2):
                for atom in range(data.particles.count):
                    if(pos_all[atom,index]<data.cell[index,index]/2):
                                #print(pid[atom],pos_all[atom,index])
                                pos_all[atom,index]+=data.cell[index,index]
                                #print(pid[atom],pos_all[atom,index])
        pos_He = data.particles["Position"][ptype==2][0]
        pos_center=pos_He #the center position of the G1 G2
        pos_around = data.particles["Position"][ptype==1] #the positions around the center      
        g11=0
        g12=0
        g2_plus1=0
        g2_plus2=0
        g2_minus1=0
        g2_minus2=0
        for i in range(np.count_nonzero(ptype==1)):
            g11+=fcutg1(R0=pos_center,Ri=pos_around[i],etaa=etaa_g1[0]) #G1
            g12+=fcutg1(R0=pos_center,Ri=pos_around[i],etaa=etaa_g1[1]) #G1
        for i in range(np.count_nonzero(ptype==1)):
            for j in range(i+1,np.count_nonzero(ptype==1)):
                    g2_plus1+=fcutg2(R0=pos_center,Ri=pos_around[i],Rj=pos_around[j],etaa=etaa_g2[0],kxi=kxi_g2[0],lambdaa=lambdaa_g2[0]) #G2
                    g2_plus2+=fcutg2(R0=pos_center,Ri=pos_around[i],Rj=pos_around[j],etaa=etaa_g2[1],kxi=kxi_g2[1],lambdaa=lambdaa_g2[0]) #G2
                    g2_minus1+=fcutg2(R0=pos_center,Ri=pos_around[i],Rj=pos_around[j],etaa=etaa_g2[0],kxi=kxi_g2[0],lambdaa=lambdaa_g2[1]) #G2
                    g2_minus2+=fcutg2(R0=pos_center,Ri=pos_around[i],Rj=pos_around[j],etaa=etaa_g2[1],kxi=kxi_g2[1],lambdaa=lambdaa_g2[1]) #G2
        G[nframe,0]=data.attributes["Timestep"]
        G[nframe,1]=g11
        G[nframe,2]=g12
        G[nframe,3]=g2_plus1
        G[nframe,4]=g2_plus2
        G[nframe,5]=g2_minus1
        G[nframe,6]=g2_minus2
    #np.savetxt("{}/G.{}.txt".format(location,batch),G)
    with open('{}/G.{}.{}.txt'.format(location,temper, num),'a') as f:
        timerange =  G.shape[0] if batch == 1000 else G.shape[0]-1
        for i in range(timerange): #Don't export the last data, to avoid the repetition
            print("{} {} {} {} {} {} {}".format(int(G[i,0]), G[i,1], G[i,2], G[i,3], G[i,4], G[i,5], G[i,6]), file=f)
    f.close()
    print(time.time()-t0)

#print("done")
