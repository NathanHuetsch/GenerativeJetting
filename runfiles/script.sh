#PBS -q a30
#PBS -l nodes=1:ppn=1:gpus=1:a30
#PBS -l walltime=40:00:00
#PBS -d /remote/gpu07/huetsch
#PBS -o output.txt
#PBS -e error.txt

# Copy-Pasted command from the qsub wiki page. Enables the job to see the cluster GPUs
export CUDA_VISIBLE_DEVICES=$(cat $PBS_GPUFILE | sed s/.*-gpu// )
export PYTHONPATH=$PYTHONPATH:/remote/gpu07/huetsch/lib/python3.9/site-packages


# Activate the python venv environment
#source venv/bin/activate
module load anaconda/3.0
module load cuda/11.7
# cd into the project folder
cd GenerativeJetting

# Run the actual python script with necessary parameters
python run_classifier.py "/remote/gpu07/huetsch/GenerativeJetting/params/classifier.yaml"

