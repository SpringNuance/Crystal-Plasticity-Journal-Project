#!/bin/bash -l
# created: Oct 19, 2022 22:22 PM
# author: xuanbinh
#SBATCH --account=project_2004956
#SBATCH --partition=large
#SBATCH --time=00:20:00
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
export PETSC_DIR=/scratch/project_2004956/userspack/install_tree/gcc-11.3.0/petsc-3.16.3-osxgr4
export PETSC_FC_INCLUDES=/scratch/project_2004956/userspack/install_tree/gcc-11.3.0/petsc-3.16.3-osxgr4/include
export PATH=/scratch/project_2004956/damask-2.0.3/damask-2.0.3/bin:/appl/soft/ai/tykky/python-data-2022-09/bin:/appl/spack/v018/install-tree/gcc-11.3.0/fftw-3.3.10-ug4bi5/bin:/appl/spack/v018/install-tree/gcc-11.3.0/openmpi-4.1.4-w2aekq/bin:/appl/spack/v018/install-tree/gcc-8.5.0/gcc-11.3.0-i44hho/bin:/appl/opt/csc-cli-utils/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/appl/bin:/users/nguyenb5/.local/bin/appl/spack/v018/install-tree/intel-2021.6.0/hdf5-1.12.2-rzlc7b/lib:/appl/spack/v018/install-tree/gcc-11.3.0/hdf5-1.12.2-t4mnjw/bin:/appl/spack/v018/install-tree/gcc-11.3.0/hdf5-1.12.2-t4mnjw/lib:/appl/spack/v018/install-tree/intel-2021.6.0/hdf5-1.12.2-rzlc7b/bin:/appl/spack/v018/install-tree/intel-2021.6.0/hdf5-1.12.2-rzlc7b/lib:/appl/spack/v018/install-tree/oneapi-2022.1.0/hdf5-1.12.2-nksn3n/lib:/appl/spack/v018/install-tree/oneapi-2022.1.0/hdf5-1.12.2-nksn3n/bin:
export LD_LIBRARY_PATH=/scratch/project_2004956/userspack/install_tree/gcc-11.3.0/petsc-3.16.3-osxgr4/lib:/appl/spack/v018/install-tree/gcc-11.3.0/fftw-3.3.10-ug4bi5/lib:/appl/spack/v018/install-tree/gcc-11.3.0/openmpi-4.1.4-w2aekq/lib:/appl/spack/v018/install-tree/gcc-8.5.0/gcc-11.3.0-i44hho/lib64:/appl/spack/v018/install-tree/gcc-8.5.0/gcc-11.3.0-i44hho/lib:/appl/spack/v018/install-tree/intel-2021.6.0/hdf5-1.12.2-rzlc7b/lib:/appl/spack/v018/install-tree/gcc-11.3.0/hdf5-1.12.2-t4mnjw/lib:/appl/spack/v018/install-tree/oneapi-2022.1.0/hdf5-1.12.2-nksn3n/lib:
export DAMASK_ROOT=/scratch/project_2004956/damask-2.0.3/damask-2.0.3

ulimit -s unlimited 
cd /scratch/project_2004956/Binh/PH_Linear
srun -n 8 DAMASK_spectral --load tensionX.load --geom 512grains512.geom
