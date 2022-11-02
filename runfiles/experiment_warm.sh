#PBS -q gshort
#PBS -l nodes=1:ppn=1:gpus=1:gshort
#PBS -l walltime=40:00:00
#PBS -d /remote/gpu07/huetsch

export CUDA_VISIBLE_DEVICES=$(cat $PBS_GPUFILE | sed s/.*-gpu// )

source venv/bin/activate
cd GenerativeJetting

python run_Z2.py --warm_start_path="/remote/gpu07/huetsch/Diffusion/runs/z2/LongerAndBigger_TBD3038" --train=False --n_samples=1000000

