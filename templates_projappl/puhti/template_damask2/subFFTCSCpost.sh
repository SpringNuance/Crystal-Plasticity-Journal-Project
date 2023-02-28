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

