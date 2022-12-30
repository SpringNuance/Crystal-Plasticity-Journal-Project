#!/bin/bash -l
# created: Feb 14, 2020 2:22 PM
# author: xuanbinh
#SBATCH --account=project_2004956
#SBATCH --partition=large
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=5
#SBATCH --error=%j_damask.err
#SBATCH --output=%j_damask.out
#SBATCH -J CPparameter_test
#SBATCH -e CPparameter_test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=binh.nguyen@aalto.fi
 
### Since postprocessing does not used DAMASK_spectral, this script cannot make use of MPI. 

### Prevent stack overflow for large models, especially when using openMP
ulimit -s unlimited 

### Enabling environments
source /scratch/project_2004956/damask-2.0.3/damask-2.0.3/damask_env.txt
source /scratch/project_2004956/damask-2.0.3/damask-2.0.3/env/DAMASK.sh
PATH=$PATH:/scratch/project_2004956/damask-2.0.3/damask-2.0.3/bin:/scratch/project_2004956/damask-2.0.3/damask-2.0.3/processing/post

### Arguments from project code
material=$1
fullpath=$(sed -n ${SLURM_ARRAY_TASK_ID}p linux_slurm/array_file.txt) 

### Changing to the current working directory
cd ${fullpath}

### Postprocessing results from the spectralOut file
postResults.py --cr texture,f,p --time ${material}_tensionX.spectralOut

### Change to the current working directory of postProc
cd ${fullpath}/postProc

### Adding additional data from postprocessed results
addCauchy.py ${material}_tensionX.txt
addStrainTensors.py --left --logarithmic ${material}_tensionX.txt
addMises.py -s Cauchy ${material}_tensionX.txt
addMises.py -e 'ln(V)' ${material}_tensionX.txt
