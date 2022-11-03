class Experiment:
    """
    Base Class for generative modelling experiment classes to inherit from.
    Children classes should overwrite the individual methods as needed.
    Depending on the details, some methods might not be needed, e.g. if the dataset is already preprocessed
    or if we just want to generate samples and plots without retraining a model.

    See z2.py for an example of a fully implemented ExperimentClass

    Structure:

    __init__(params)      : Read in parameters
    prepare_experiment()  : Create out_dir and cwd into chdir into it
    load_data()           : Read in the dataset
    preprocess_data()     : Preprocess the data and move it to device
    build_model()         : Build the model and define its architecture
    build_optimizer()     : Build the optimizer and define its parameters
    build_dataloaders()   : Build the dataloaders for model training
    train_model()         : Run the model training
    generate_samples()    : Use the model to generate samples
    make_plots()          : Make plots to compare the generated samples to the test set data
    finish_up()           : Finish the experiment and save some information

    full_run()            : Call all of the above methods to perform a full experiment
    """

    def __init__(self, params):
        pass

    def prepare_experiment(self):
        pass

    def load_data(self):
        pass

    def preprocess_data(self):
        pass

    def build_model(self):
        pass

    def build_optimizer(self):
        pass

    def build_dataloaders(self):
        pass

    def train_model(self):
        pass

    def generate_samples(self):
        pass

    def make_plots(self):
        pass

    def finish_up(self):
        pass

    def full_run(self):
        """
        Performs one complete experiment
        """
        self.prepare_experiment()
        self.load_data()
        self.preprocess_data()
        self.build_model()
        self.build_optimizer()
        self.build_dataloaders()
        self.train_model()
        self.generate_samples()
        self.make_plots()
        self.finish_up()
