#PBS -q gshort
#PBS -l nodes=1:ppn=1:gpus=1:gshort
#PBS -l walltime=40:00:00
#PBS -d /remote/gpu07/huetsch

export CUDA_VISIBLE_DEVICES=$(cat $PBS_GPUFILE | sed s/.*-gpu// )
export PYTHONPATH=$PYTHONPATH:/remote/gpu07/huetsch/lib/python3.9/site-packages


# Activate the python venv environment
#source venv/bin/activate
module load anaconda/3.0
module load cuda/11.7
# cd into the project folder
cd GenerativeJetting

python run_toy.py --warm_start_path="/remote/gpu07/huetsch/GenerativeJetting/runs/Toy/OdeTesting/Bayes/run_06635" --train=False --redirect_console=False --plot=True

