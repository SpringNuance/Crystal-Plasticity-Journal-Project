#!/bin/bash -l
# created: 13.07.2021
# author: WenqiLiu
#SBATCH --account=project_2004956
#SBATCH --partition=small
#SBATCH --time=1:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH -J Array_CPPost
#SBATCH -e Array_CPPost
#SBATCH --mail-type=ALL
#SBATCH --mail-user=binh.nguyen@aalto.fi
 
### Change to the work directory
cd /scratch/project_2004956/Binh/CrystalPlasticityProject
ulimit -s unlimited 
source /projappl/project_2001353/Zinan/damask2.0.2/DAMASK/DAMASK_env.sh
PATH=$PATH:/projappl/project_2001353/Zinan/damask2.0.2/DAMASK/processing/post

postResults.py --cr f,p,texture --time RVE_1_40_D_tensionX.spectralOut
cd /scratch/project_2004956/Binh/CrystalPlasticityProject/template/postProc
addCauchy.py RVE_1_40_D_tensionX.txt
addStrainTensors.py --left --logarithmic RVE_1_40_D_tensionX.txt
addMises.py -s Cauchy RVE_1_40_D_tensionX.txt
addMises.py -e 'ln(V)' RVE_1_40_D_tensionX.txt
