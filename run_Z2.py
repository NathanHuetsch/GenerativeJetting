from Source.Experiments import z2
from Source.Util.util import load_params
from absl import app
from absl import flags
import sys
import os
import warnings
warnings.filterwarnings("ignore")

"""
The *.py script that should be called from the *.sh shell script.
There are several ways to call this script:

1) The Standard way 
Call this function and pass as single argument the path to the *.yaml file containing the experiment parameters.
No flags have to be specified and the experiment will just run from the parameter *.yaml.
Example:
python run_Z2.py /remote/gpu07/huetsch/GenerativeJetting/params/Z2_Experiment_gpu.yaml

2) The WarmStart way
Call this function as pass a path to a warm start folder as warm_start_path flag. 
In this case we do not need a path to a *.yaml parameter file. It is assumed that the warm start folder contains a
file called "parameters.yaml". These will be read and used for the experiment.
This is convenient if we want to continue a past experiment, as we will just load the parameters from that experiment.
Example:
python run_Z2.py --warm_start_path="/remote/gpu07/huetsch/GenerativeJetting/runs/z2/LongerAndBigger_TBD3038"

3) The Overwrite way
Can be combined with 2)
Pass additional flags to overwrite specific parameters in the *.yaml parameter file. This is useful if we want to
continue a past experiment, but change some of the parameters, but are too lazy do to it by hand.
(Note: Not fully implemented for arbitray parameters yet)

The most common use case for this is if we want to load a pretrained model and generate new samples and/or plots with it.
In this case we can load the past experiment with 2) and overwrite the train parameter in it. 
Example:
python run_Z2.py --warm_start_path="/remote/gpu07/huetsch/GenerativeJetting/runs/z2/LongerAndBigger_TBD3038" --train=False --n_samples=1000000
"""


def define_flags():
    flags.DEFINE_string('warm_start_path', None, "Path to the pre-trained model folder")
    flags.DEFINE_string('overwrite_param_path', None, "Path to another param file to overwrite some params")
    flags.DEFINE_boolean('train', None, "Overwrite train parameter in params")
    flags.DEFINE_boolean('sample', None, "Overwrite sample parameter in params")
    flags.DEFINE_integer('n_samples', None, "Overwrite n_samples in params")
    flags.DEFINE_boolean('plot', None, "Overwrite plot parameter in params")


def main(argv):

    if FLAGS.warm_start_path is None:
        params = load_params(sys.argv)
    else:
        params = load_params(os.path.join(FLAGS.warm_start_path, "paramfile.yaml"))
        params["warm_start"] = True
        params["warm_start_path"] = FLAGS.warm_start_path

    if FLAGS.overwrite_param_path is not None:
        overwrite_params = load_params(FLAGS.overwrite_param_path)
        params = params | overwrite_params

    if FLAGS.train is not None:
        params["train"] = FLAGS.train

    if FLAGS.sample is not None:
        params["sample"] = FLAGS.sample

    if FLAGS.n_samples is not None:
        params["n_samples"] = FLAGS.n_samples

    if FLAGS.plot is not None:
        params["plot"] = FLAGS.plot

    experiment = z2.Z2_Experiment(params)
    experiment.full_run()


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    define_flags()
    app.run(main)
