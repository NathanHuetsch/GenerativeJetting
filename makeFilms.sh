#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/remote/gpu07/huetsch/lib/python3.9/site-packages
export PYTHONPATH=$PYTHONPATH:/remote/gpu07/huetsch/GenerativeJetting
module load anaconda/3.0
module load cuda/11.7
python mu_sigma_plot.py $1
python film.py $1/mu_sigma
