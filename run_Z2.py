from Source.Experiments import z2
from Source.Util.util import load_params
from absl import app
from absl import flags
import sys
import os
import warnings
warnings.filterwarnings("ignore")


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
