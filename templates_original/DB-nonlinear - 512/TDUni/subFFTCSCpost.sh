#!/bin/bash -l
# created: 13.07.2021
# author: WenqiLiu
#SBATCH --account=project_2004956
#SBATCH --partition=small
#SBATCH --time=71:00:00
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH -J CPPost
#SBATCH -e CPPost
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rongfei.juan@aalto.fi

### Change to the work directory
cd /scratch/project_2001353/Rongfei/Binh/DBshape1

ulimit -s unlimited

source /projappl/project_2004956/DAMASK/damask2.0.2/DAMASK/DAMASK_env.sh
PATH=$PATH:/projappl/project_2004956/DAMASK/damask2.0.2/DAMASK/processing/post
postResults.py --cr texture,f,p --time 512grains512_tensionX.spectralOut
cd /scratch/project_2001353/Rongfei/Binh/DBshape1/postProc

addCauchy.py 512grains512_tensionX.txt
addStrainTensors.py --left --logarithmic 512grains512_tensionX.txt
addMises.py -s Cauchy 512grains512_tensionX.txt
addMises.py -e 'ln(V)' 512grains512_tensionX.txt
