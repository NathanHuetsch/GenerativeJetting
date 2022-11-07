# File to call with qsub to run experiment on the GPU cluster

# Make some specifications:
# -q determines the queue used
# -l determines the required computational ressources
# -d determines the directory to jump into

#PBS -q gshort
#PBS -l nodes=1:ppn=1:gpus=1:gshort
#PBS -l walltime=40:00:00
#PBS -d /remote/gpu07/huetsch

# Copy-Pasted command from the qsub wiki page. Enables the job to see the cluster GPUs
export CUDA_VISIBLE_DEVICES=$(cat $PBS_GPUFILE | sed s/.*-gpu// )

# Activate the python venv environment
source venv/bin/activate
# cd into the project folder
cd GenerativeJetting

# Run the actual python script with necessary parameters
python run_Z2.py /remote/gpu07/huetsch/GenerativeJetting/params/Z26c_DDPM.yaml

