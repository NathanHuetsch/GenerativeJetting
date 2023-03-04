#!/bin/bash

source ../venv/bin/activate
python mu_sigma_plot.py $1
python film.py $1/mu_sigma
