#!/bin/bash
input=(${HQ_ENTRY})
fullpath=${input[0]}
material=${input[1]}

cd ${fullpath}

orterun DAMASK_spectral --load tensionX.load --geom ${material}.geom