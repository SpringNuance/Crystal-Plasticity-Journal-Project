#!/bin/bash -l
# created: Feb 14, 2020 2:22 PM
# author: xuanbinh
#SBATCH --account=project_2004956
#SBATCH --partition=large
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --error=%j_damask.err
#SBATCH --output=%j_damask.out
#SBATCH -J CPparameter_test
#SBATCH -e CPparameter_test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=binh.nguyen@aalto.fi
 
### Change to the work directory
cd /scratch/project_2004956/Binh/PH_Linear
ulimit -s unlimited 
source /scratch/project_2004956/damask-2.0.3/damask-2.0.3/damask_env.txt
source /scratch/project_2004956/damask-2.0.3/damask-2.0.3/env/DAMASK.sh
PATH=$PATH:/scratch/project_2004956/damask-2.0.3/damask-2.0.3/bin:/scratch/project_2004956/damask-2.0.3/damask-2.0.3/processing/post

postResults.py --cr f,p,texture --time RVE_1_40_D_tensionX.spectralOut
cd /scratch/project_2004956/Binh/PH_Linear/postProc
addCauchy.py RVE_1_40_D_tensionX.txt
addStrainTensors.py --left --logarithmic RVE_1_40_D_tensionX.txt
addMises.py -s Cauchy RVE_1_40_D_tensionX.txt
addMises.py -e 'ln(V)' RVE_1_40_D_tensionX.txt
