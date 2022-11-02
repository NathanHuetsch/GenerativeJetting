#PBS -q gshort
#PBS -l nodes=1:ppn=1:gpus=1:gshort
#PBS -l walltime=40:00:00
#PBS -d /remote/gpu07/huetsch

export CUDA_VISIBLE_DEVICES=$(cat $PBS_GPUFILE | sed s/.*-gpu// )

source venv/bin/activate
cd Diffusion

python main.py /remote/gpu07/huetsch/Diffusion/params/DDPM_6c.yaml

