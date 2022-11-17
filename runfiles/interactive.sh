#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=40:00:00
#PBS -d /remote/gpu07/huetsch
#PBS -o output.txt
#PBS -e error.txt

export CUDA_VISIBLE_DEVICES=$(cat $PBS_GPUFILE | sed s/.*-gpu// )
export PYTHONPATH=$PYTHONPATH:/remote/gpu07/huetsch/lib/python3.9/site-packages

