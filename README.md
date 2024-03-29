# GenerativeJetting
Python code to perform generative ML modelling of jet data.

The idea is to be able to conveniently run new experiments, continue old experiments, change parameters, generate new samples and/or plots.
Most of the boilerplate code is hidden behind the ExperimentClass. It takes a parameter dictionary as input and then takes care of everything.

Sofia war hier.

The code is written to run on our GPU cluster and hierarchically works as follows:

- Experiments are initiated from a *.sh file in the runfiles folder. This should follow the syntax of the GPU cluster qsub system. This file should then call upon a main method in the main folder and pass the necessary parameters and flags, most importantly a *.yaml file including the parameters defining the experiment.
  Example: runfiles/experiment.sh

- The next step is a main *.py file in the main folder to be called upon from the *.sh file. This file should read in and process the parameters and flags, create an experiment and run it.
  Example: run_Z2.py

- Experiments are performed using the Experiment classes in the Source/Experiments folder. These classes should implement all necessary functions to perform the experiments. It is recommended to inherit from the Experiment base class defined in Source/Experiments/ExperimentBase.py and follow the defined structure.
But feel free to implement other types of experiment :)
  Example: Source/Experiments/z2.py

- Generative models are instantiated within the Experiment classes. Models should be implemented in the Source/Models folder and should inherit from and follow the structure of Source/Models/ModelBase.py
  Example: Source/Models/tbd.py

- Neural Netowkrs are instantiated within the Models classes. New networks should be implemented in the Souce/Networks folder. They do not have to follow any specific structure, the only requirement is that they are instantiated with a parameter dict as single input.
  Example: Source/Networks/resnet.py

Detailed documentation for the classes and their methods can be found in the code.
The principal steps of an experiment are outlined in the Experiment base class in Source/Experiments/ExperimentBase.py
