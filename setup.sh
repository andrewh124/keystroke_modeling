#!/bin/bash
#$ -cwd
conda create -y --name $CONDA_ENV python=3.10
conda activate $CONDA_ENV
conda install -y pytorch==1.12.1 cudatoolkit=11.6 -c pytorch
conda install -y --file requirements.txt