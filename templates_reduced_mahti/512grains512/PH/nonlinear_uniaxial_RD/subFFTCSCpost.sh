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
#SBATCH --mail-user=<your registered email>

### Since postprocessing does not used DAMASK_spectral, this script cannot make use of MPI. 
 
### !Fix this line. Change this to your current working directory
cd <current working directory of this template>

### Prevent stack overflow for large models, especially when using openMP
ulimit -s unlimited 

### Enabling environments
source /scratch/project_2004956/damask-2.0.3/damask-2.0.3/damask_env.txt
source /scratch/project_2004956/damask-2.0.3/damask-2.0.3/env/DAMASK.sh
PATH=$PATH:/scratch/project_2004956/damask-2.0.3/damask-2.0.3/bin:/scratch/project_2004956/damask-2.0.3/damask-2.0.3/processing/post

### Postprocessing results from the spectralOut file
postResults.py --cr f,p,texture --time 512grains512_tensionX.spectralOut

### !Fix this line. Change this to your current working directory
cd <current working directory of this template>/postProc

### Adding additional data from postprocessed results
addCauchy.py 512grains512_tensionX.txt
addStrainTensors.py --left --logarithmic 512grains512_tensionX.txt
addMises.py -s Cauchy 512grains512_tensionX.txt
addMises.py -e 'ln(V)' 512grains512_tensionX.txt
