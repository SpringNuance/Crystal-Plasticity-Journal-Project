#!/bin/bash -l
# created: Oct 19, 2022 22:22 PM
# author: xuanbinh
#SBATCH --account=project_2004956
#SBATCH --partition=large
#SBATCH --time=02:00:00
#SBACTH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=5
#SBATCH -J CPparameter_test
#SBATCH -e CPparameter_test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=binh.nguyen@aalto.fi

### In Puhti, the partition can be large or small. Large partition runs faster
### The number of threads in a node in Puhti is 40
### ntasks-per-node is the number of parallel tasks that one node can run
### cpus-per-task is the number of threads that can work on each task
### You must ensure that (ntasks-per-node * cpus-per-task = number of threads)
### For example, 8 x 5 = 40

# This line is important for DAMASK to utilize all threads based on the cpus-per-task 
# In DAMASK 3, it should be OMP_NUM_THREADS instead of DAMASK_NUM_THREADS
export DAMASK_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Setting up PETSc and DAMASK working directories in Puhti server
export PETSC_DIR=/projappl/project_2004956/spack/install_tree/gcc-11.3.0/petsc-3.16.3-xle26t
export PETSC_FC_INCLUDES=/projappl/project_2004956/spack/install_tree/gcc-11.3.0/petsc-3.16.3-xle26t/include
export PATH=$PATH:/projappl/project_2004956/damask2/damask-2.0.3/bin:/appl/soft/ai/tykky/python-data-2022-09/bin:/appl/spack/v018/install-tree/gcc-11.3.0/fftw-3.3.10-ug4bi5/bin:/appl/spack/v018/install-tree/gcc-11.3.0/openmpi-4.1.4-w2aekq/bin:/appl/spack/v018/install-tree/gcc-8.5.0/gcc-11.3.0-i44hho/bin:/appl/opt/csc-cli-utils/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/appl/bin:/users/nguyenb5/.local/bin/appl/spack/v018/install-tree/intel-2021.6.0/hdf5-1.12.2-rzlc7b/lib:/appl/spack/v018/install-tree/gcc-11.3.0/hdf5-1.12.2-t4mnjw/bin:/appl/spack/v018/install-tree/gcc-11.3.0/hdf5-1.12.2-t4mnjw/lib:/appl/spack/v018/install-tree/intel-2021.6.0/hdf5-1.12.2-rzlc7b/bin:/appl/spack/v018/install-tree/intel-2021.6.0/hdf5-1.12.2-rzlc7b/lib:/appl/spack/v018/install-tree/oneapi-2022.1.0/hdf5-1.12.2-nksn3n/lib:/appl/spack/v018/install-tree/oneapi-2022.1.0/hdf5-1.12.2-nksn3n/bin:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projappl/project_2004956/spack/install_tree/gcc-11.3.0/petsc-3.16.3-xle26t/lib:/appl/spack/v018/install-tree/gcc-11.3.0/fftw-3.3.10-ug4bi5/lib:/appl/spack/v018/install-tree/gcc-11.3.0/openmpi-4.1.4-w2aekq/lib:/appl/spack/v018/install-tree/gcc-8.5.0/gcc-11.3.0-i44hho/lib64:/appl/spack/v018/install-tree/gcc-8.5.0/gcc-11.3.0-i44hho/lib:/appl/spack/v018/install-tree/intel-2021.6.0/hdf5-1.12.2-rzlc7b/lib:/appl/spack/v018/install-tree/gcc-11.3.0/hdf5-1.12.2-t4mnjw/lib:/appl/spack/v018/install-tree/oneapi-2022.1.0/hdf5-1.12.2-nksn3n/lib:
export DAMASK_ROOT=/projappl/project_2004956/damask2/damask-2.0.3

### Prevent stack overflow for large models, especially when using openMP
ulimit -s unlimited 

### Arguments from project code
material=$1
curvetype=$2

fullpath=$(sed -n ${SLURM_ARRAY_TASK_ID}p linux_slurm/array_${curvetype}_file.txt) 

### Change to the current working directory
cd ${fullpath}

### MPI run by the DAMASK_spectral software
### Important!: The number of nodes in srun -n <number> must be equal to --ntasks
srun -n 8 DAMASK_spectral --load tensionX.load --geom ${material}.geom