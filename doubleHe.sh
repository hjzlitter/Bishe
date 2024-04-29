#!/bin/sh
#An example for MPI job.
#SBATCH -J doubleHe
#SBATCH -o job-%j.log
#SBATCH -e job-%j.err
#SBATCH -N 1 -n 20
#SBATCH -p GPU-V100
#SBATCH --qos=gpujoblimit --time=10-00
echo Time is `date`
echo Directory is $PWD
echo This job runs on the following nodes:
echo $SLURM_JOB_NODELIST
echo This job has allocated $SLURM_JOB_CPUS_PER_NODE cpu cores.

MPIRUN=mpirun #Intel mpi and Open MPI
MPIOPT=

module load lammps/3Mar20
export PATH=$PATH:/home/snst/huysh20/.lammps_command
export LAMMPS_POTENTIALS=/home/snst/huysh20/.lammps_command

$MPIRUN lmp_gpu_200303 -sf gpu -pk gpu 2 -in in.doubleHe.py

echo Time is `date`


#!/bin/sh
#An example for MPI job.
#SBATCH -J Annealing
#SBATCH -o job-%j.log
#SBATCH -e job-%j.err
#SBATCH -N 1 -n 20
#SBATCH -p GPU-V100
#SBATCH --gres=gpu:2
#SBATCH --qos=gpujoblimit --time=10-00
echo Time is `date`
echo Directory is $PWD
echo This job runs on the following nodes:
echo $SLURM_JOB_NODELIST
echo This job has allocated $SLURM_JOB_CPUS_PER_NODE cpu cores.

MPIRUN=mpirun #Intel mpi and Open MPI
MPIOPT=

module load lammps/3Mar20
export PATH=$PATH:/home/snst/huysh20/.lammps_command
export LAMMPS_POTENTIALS=/home/snst/huysh20/.lammps_command

$MPIRUN lmp_gpu_200303 -sf gpu -pk gpu 2 -in in.doubleHe.py

echo Time is `date`

