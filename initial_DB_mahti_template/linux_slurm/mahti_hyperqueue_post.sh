#!/bin/bash
#SBATCH --account=project_2004956
#SBATCH --partition=large
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --cpus-per-task=40
#SBATCH --ntasks-per-node=1
#SBATCH -J CPparameter_test
#SBATCH -e CPparameter_test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=binh.nguyen@aalto.fi

export SLURM_EXACT=1

# Load the required modules
module --force purge
module load hyperqueue
module load gcc/11.3.0
module load openmpi/4.1.4
module load hdf5/1.12.2-mpi
module load fftw/3.3.10-mpi
module load python-data/3.10-22.09

# Specify with how many threads you want to run each subtask
export DAMASK_NUM_THREADS=16

# Set needed environment variables
export PETSC_DIR=/scratch/project_2004956/userspack/install_tree/gcc-11.2.0/petsc-3.16.1-zeqfqr
export PETSC_FC_INCLUDES=/scratch/project_2004956/userspack/install_tree/gcc-11.2.0/petsc-3.16.1-zeqfqr/include
export PATH=$PATH:/scratch/project_2004956/damask-2.0.3/damask-2.0.3/bin:/appl/soft/ai/tykky/python-data-2022-09/bin:/appl/spack/v018/install-tree/gcc-11.3.0/fftw-3.3.10-ug4bi5/bin:/appl/spack/v018/install-tree/gcc-11.3.0/openmpi-4.1.4-w2aekq/bin:/appl/spack/v018/install-tree/gcc-8.5.0/gcc-11.3.0-i44hho/bin:/appl/opt/csc-cli-utils/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/appl/bin:/users/nguyenb5/.local/bin/appl/spack/v018/install-tree/intel-2021.6.0/hdf5-1.12.2-rzlc7b/lib:/appl/spack/v018/install-tree/gcc-11.3.0/hdf5-1.12.2-t4mnjw/bin:/appl/spack/v018/install-tree/gcc-11.3.0/hdf5-1.12.2-t4mnjw/lib:/appl/spack/v018/install-tree/intel-2021.6.0/hdf5-1.12.2-rzlc7b/bin:/appl/spack/v018/install-tree/intel-2021.6.0/hdf5-1.12.2-rzlc7b/lib:/appl/spack/v018/install-tree/oneapi-2022.1.0/hdf5-1.12.2-nksn3n/lib:/appl/spack/v018/install-tree/oneapi-2022.1.0/hdf5-1.12.2-nksn3n/bin:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/project_2004956/userspack/install_tree/gcc-11.2.0/petsc-3.16.1-zeqfqr/lib:/appl/spack/v018/install-tree/gcc-11.3.0/fftw-3.3.10-ug4bi5/lib:/appl/spack/v018/install-tree/gcc-11.3.0/openmpi-4.1.4-w2aekq/lib:/appl/spack/v018/install-tree/gcc-8.5.0/gcc-11.3.0-i44hho/lib64:/appl/spack/v018/install-tree/gcc-8.5.0/gcc-11.3.0-i44hho/lib:/appl/spack/v018/install-tree/intel-2021.6.0/hdf5-1.12.2-rzlc7b/lib:/appl/spack/v018/install-tree/gcc-11.3.0/hdf5-1.12.2-t4mnjw/lib:/appl/spack/v018/install-tree/oneapi-2022.1.0/hdf5-1.12.2-nksn3n/lib:
export DAMASK_ROOT=/scratch/project_2004956/damask-2.0.3/damask-2.0.3

# Set the directory which hyperqueue will use
export HQ_SERVER_DIR=$PWD/hq-server-$SLURM_JOB_ID
mkdir -p "$HQ_SERVER_DIR"

echo "STARTING HQ SERVER, log in $HQ_SERVER_DIR/HQ.log"
echo "===================="
hq server start &>> "$HQ_SERVER_DIR/HQ.log" &
until hq job list &>/dev/null ; do sleep 1 ; done

echo "STARTING HQ WORKERS ON $SLURM_NNODES nodes"
echo "===================="
srun --cpu-bind=none --mpi=none hq worker start --cpus=$SLURM_CPUS_PER_TASK &>> "$HQ_SERVER_DIR/HQ.log" &

until [[ $num_up -eq $SLURM_NNODES ]]; do
    num_up=$(hq worker list 2>/dev/null | grep -c RUNNING)
    echo "WAITING FOR WORKERS TO START ( $num_up / $SLURM_NNODES )"
    sleep 1
done

# This is the important part, here you submit your script as a task array
# A list of filenames to be processed is provided in tasklist.txt

hq submit --cpus $DAMASK_NUM_THREADS --each-line linux_slurm/hyperqueue_file.txt linux_slurm/mahti_task_post.sh
# Array version
# hq submit --cpus $DAMASK_NUM_THREADS --array=1-100 damask.sh

while hq job list --all | grep -q "RUNNING\|PENDING"; do
    echo "WAITING FOR JOBS TO FINISH"
    # Adjust the timing here if you get to much output in the Slurm log file
    # Now set to 30 seconds
    sleep 30
done

echo "===================="
echo "DONE"
echo "===================="
echo "SHUTTING DOWN HYPERQUEUE"
echo "===================="
hq worker stop all
hq server stop