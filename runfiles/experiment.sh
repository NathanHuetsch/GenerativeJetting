#!/bin/bash                                                                     
#PBS -l walltime=30:00:00                                                        
#PBS -l nodes=1:ppn=1:gpus=1:a30                                        
#PBS -q a30                                                                 

module load cuda/12.1
export PATH=/remote/gpu03/anaconda3/bin:$PATH
source activate /remote/gpu03/hoelzl/conda/venv

mydev=`cat $PBS_GPUFILE | sed s/.*-gpu// `
export CUDA_VISIBLE_DEVICES=$mydev

cd /remote/gpu03/hoelzl/scripts/GenerativeJetting

nice -19 python run_Zn.py /remote/gpu03/hoelzl/scripts/GenerativeJetting/params/gpu_test.yaml

echo "job done"
