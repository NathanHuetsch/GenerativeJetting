#PBS -q gshort
#PBS -l nodes=1:ppn=1:gpus=1:gshort
#PBS -l walltime=40:00:00
#PBS -d /remote/gpu05/palacios

export CUDA_VISIBLE_DEVICES=$(cat $PBS_GPUFILE | sed s/.*-gpu// )

source venv/bin/activate
cd GenerativeJetting

python run_Zn.py --warm_start_path="/remote/gpu05/palacios/GenerativeJetting/runs/z213/TBD_att_full5452"

