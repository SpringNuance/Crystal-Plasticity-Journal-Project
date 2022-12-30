#!/bin/bash -l
# created: Oct 19, 2022 22:22 PM
# author: xuanbinh
#SBATCH --account=project_2004956
#SBATCH --partition=medium
#SBATCH --time=01:00:00
#SBACTH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --hint=multithread
#SBATCH -J CPparameter_test
#SBATCH -e CPparameter_test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=binh.nguyen@aalto.fi

### Since postprocessing does not used DAMASK_spectral, this script cannot make use of MPI. 
 
### Change to your current working directory
cd $PWD

### Prevent stack overflow for large models, especially when using openMP
ulimit -s unlimited 

### Enabling environments
source /projappl/project_2004956/damask2/damask_env.txt
source /projappl/project_2004956/damask2/damask-2.0.3/env/DAMASK.sh
PATH=$PATH:/projappl/project_2004956/damask2/damask-2.0.3/bin:/projappl/project_2004956/damask2/damask-2.0.3/processing/post

### Postprocessing results from the spectralOut file
postResults.py --cr f,p,texture --time RVE_1_40_D_tensionX.spectralOut

### Change to your current working directory
cd $PWD/postProc

### Adding additional data from postprocessed results
addCauchy.py RVE_1_40_D_tensionX.txt
addStrainTensors.py --left --logarithmic RVE_1_40_D_tensionX.txt
addMises.py -s Cauchy RVE_1_40_D_tensionX.txt
addMises.py -e 'ln(V)' RVE_1_40_D_tensionX.txt
