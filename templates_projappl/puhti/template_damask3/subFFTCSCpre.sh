#!/bin/bash -l
# created: Feb 14, 2020 2:22 PM
# author: xuanbinh
#SBATCH --account=project_2004956
#SBATCH --partition=large
#SBATCH --time=01:00:00
#SBACTH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=5
#SBATCH -J CPparameter_test
#SBATCH -e CPparameter_test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=binh.nguyen@aalto.fi
 
module load gcc/11.3.0
module load openmpi/4.1.4
module load hdf5/1.12.2-mpi
module load fftw/3.3.10-mpi
module load python-data/3.10-22.09

export DAMASK_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PETSC_DIR=/projappl/project_2004956/spack/install_tree/gcc-11.3.0/petsc-3.16.3-xle26t
export PETSC_FC_INCLUDES=/projappl/project_2004956/spack/install_tree/gcc-11.3.0/petsc-3.16.3-xle26t/include
export LD_LIBRARY_PATH=/projappl/project_2004956/spack/install_tree/gcc-11.3.0/petsc-3.16.3-xle26t/lib:$LD_LIBRARY_PATH
export PATH=/projappl/project_2004956/damask3/grid_solver/bin:$PATH
export DAMASK_ROOT=/projappl/project_2004956/damask3/damask-3.0.0-alpha6

ulimit -s unlimited

cd $PWD

srun -n 8 DAMASK_grid --load tensionX.yaml --geom RVE_1_40_D.vti
