import sys
from run import run
import numpy as np
import os
from Source.Util.util import load_params
from absl import app
from absl import flags


def define_flags():
    flags.DEFINE_boolean('warm_start', False, "Start from a pre-trained model")
    flags.DEFINE_string('warm_start_path', "", "Path to the pre-trained model folder")


def main(argv):

    if not FLAGS.warm_start:
        params = load_params(sys.argv)
        params["warm_start"] = False
        run_name = params["run_name"]
        runs_dir = params["runs_dir"]
        os.makedirs(runs_dir)
        out_dir = os.path.join(runs_dir, run_name + str(np.random.randint(low=1000, high=9999)))
        os.makedirs(out_dir)
        os.chdir(out_dir)
    else:
        assert os.path.exists(FLAGS.warm_start_path)
        paramfile_path = os.path.join(FLAGS.warm_start_path, "paramfile.yaml")
        params = load_params(paramfile_path)
        params["warm_start"] = True
        params["warm_start_path"] = FLAGS.warm_start_path
        out_dir = os.path.join(FLAGS.warm_start_path, "continue"+ str(np.random.randint(low=1000, high=9999)))
        os.makedirs(out_dir)
        os.chdir(out_dir)

    if "gpu07" in os.getcwd():
        sys.stdout = open("stdout.txt","w")
        sys.stderr = open("stderr.txt","w")
        run(params)
        sys.stdout.close()
        sys.stderr.close()
    else:
        run(params)


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    define_flags()
    app.run(main)




