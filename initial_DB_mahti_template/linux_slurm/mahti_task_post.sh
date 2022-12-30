#!/bin/bash
### Arguments from project code
input=(${HQ_ENTRY})
fullpath=${input[0]}
material=${input[1]}

### Running environments 
ulimit -s unlimited
source /scratch/project_2004956/damask-2.0.3/damask-2.0.3/damask_env.txt
source /scratch/project_2004956/damask-2.0.3/damask-2.0.3/env/DAMASK.sh
PATH=$PATH:/scratch/project_2004956/damask-2.0.3/damask-2.0.3/bin:/scratch/project_2004956/damask-2.0.3/damask-2.0.3/processing/post

### Change to the work directory
cd ${fullpath}
postResults.py --cr texture,f,p --time ${material}_tensionX.spectralOut

cd ${fullpath}/postProc
addCauchy.py ${material}_tensionX.txt
addStrainTensors.py --left --logarithmic ${material}_tensionX.txt
addMises.py -s Cauchy ${material}_tensionX.txt
addMises.py -e 'ln(V)' ${material}_tensionX.txt