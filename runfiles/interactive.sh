#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=40:00:00
#PBS -d /remote/gpu07/huetsch

export CUDA_VISIBLE_DEVICES=$(cat $PBS_GPUFILE | sed s/.*-gpu// )
# export PYTHONPATH=$PYTHONPATH:/remote/gpu07/huetsch/lib/python3.9/site-packages
# export PYTHONPATH=$PYTHONPATH:/remote/gpu07/huetsch/GenerativeJetting
# module load anaconda/3.0
# module load cuda/11.7