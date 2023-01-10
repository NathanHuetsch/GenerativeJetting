import os
from Source.Experiments import classifier_experiment
from Source.Util.util import load_params
from absl import app
import sys
import os
import warnings
warnings.filterwarnings("ignore")


def main(argv):

    params = load_params(sys.argv[1])
    experiment = classifier_experiment.Classifier_Experiment(params)
    experiment.full_run()


if __name__ == '__main__':
    app.run(main)
