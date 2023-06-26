import numpy as np
import torch
from scipy.integrate import solve_ivp
import Source.Networks
from Source.Util.util import get
from Source.Models.ModelBase import GenerativeModel


class TBD(GenerativeModel):
    """
     Class for Conditional Flow Matching
     Inheriting from the GenerativeModel BaseClass
    """

    def __init__(self, params):
        super().__init__(params)

        # Fix the diffusion trajectory. Default is linear
        trajectory = get(self.params, "trajectory", "linear_trajectory")
        try:
            self.trajectory = getattr(Source.Models.tbd, trajectory)
        except AttributeError:
            raise NotImplementedError(f"build_model: Trajectory type {trajectory} not implemented")

        # Fix the weighting for the KL loss. Default is 1
        self.C = get(self.params, "C", 1)
        if self.C != 1:
            print(f"C is {self.C}")

        # Fix the distribution to sample t from during training. Default is Uniform
        self.beta_dist = get(self.params, "beta_dist", False)
        if self.beta_dist:
            self.beta_dist_a = get(self.params, "beta_dist_a", 1)
            self.beta_dist_b = get(self.params, "beta_dist_b", 0.7)
            self.dist = torch.distributions.beta.Beta(concentration1=float(self.beta_dist_a),
                                                      concentration0=float(self.beta_dist_b))
            print(f"Using beta distribution to sample t with params {self.beta_dist_a, self.beta_dist_b}")
        else:
            self.dist = torch.distributions.uniform.Uniform(low=0, high=1)
            print(f"Using uniform distribution to sample t")

        # Bayesian or Deterministic. Default is Deterministic
        self.bayesian = get(self.params, "bayesian", 0)

        # Use magic transformation for DeltaR. Default is False
        self.magic_transformation = get(self.params, "magic_transformation", False)

    def build_net(self):
        """
        Build the network
        """
        network = get(self.params, "network", "Resnet")
        try:
            return getattr(Source.Networks, network)(self.params).to(self.device)
        except AttributeError:
            raise NotImplementedError(f"build_model: Network class {network} not recognised")

    def batch_loss(self, x):
        """
        Calculate batch loss for one batch of samples
        """

        # Calculate condition and/or weights if used
        x, condition, weights = self.get_condition_and_input(x)

        # Sample a set of timesteps
        t = self.dist.sample((x.size(0),1)).to(x.device)
        # Sample a set of noise variables
        x_0 = torch.randn_like(x)
        # Calculate the trajectory and derivative
        x_t, x_t_dot = self.trajectory(x_0, x, t)

        # Reset network state
        self.net.kl = 0
        # Predict velocity field
        drift = self.net(x_t, t, condition)

        # Calculate loss
        if self.magic_transformation:
            loss = torch.mean((drift - x_t_dot) ** 2 * weights[:, None])/weights.mean()
        else:
            loss = torch.mean((drift - x_t_dot) ** 2)
            self.regular_loss.append(loss.detach().cpu().numpy())
        if self.C != 0:
            kl_loss = self.C*self.net.kl / self.n_traindata
            self.kl_loss.append(kl_loss.detach().cpu().numpy())
            loss = loss + kl_loss
        return loss

    def get_condition_and_input(self,input):
        """
        :param input: model input + conditional input
        :return: model input, conditional input
        """
        x = input.clone()
        # if true use last entry of input as weights
        if self.magic_transformation:
            weights = x[:, -1]
            x = x[:, :-1]
        else:
            weights = None

        # get conditions for different networks
        if self.conditional and self.n_jets == 1:
            # get hot-encoded jet count (3d base) from first three entries of input
            condition = x[:, :3]
            x = x[:, 3:]

        elif self.conditional and self.n_jets == 2:
            # get first 9 entries of input corresponding to mu, mu, jet
            condition_1 = x[:, 2:11]
            # get hot-encoded jet count (2d base) from first two entries of input
            condition_2 = x[:, :2]
            # add all conditions
            condition = torch.cat([condition_1, condition_2], 1)
            # define model input to last four entries corresponding to second jet
            x = x[:, 11:]

        elif self.conditional and self.n_jets == 3:
            # get first 13 entries of input corresponding to mu, mu, jet, jet
            condition = x[:, :13]
            # define model input to last four entries corresponding to third jet
            x = x[:, 13:]


        else:
            condition = None

        return x, condition, weights

    def sample_n(self, n_samples, prior_samples=None, con_depth=0):
        """
        Generate n_samples new samples.
        Start from Gaussian random noise and solve the reverse ODE to obtain samples

        prior_samples, con_depth related to conditional generation
        """
        # If Bayesian, check if we want to sample with MAP weights
        if self.net.bayesian:
            self.net.map = get(self.params,"fix_mu", False)
            for bay_layer in self.net.bayesian_layers:
                bay_layer.random = None
        self.eval()
        batch_size = self.batch_size_sample

        # Sample latent space noise
        x_T = np.random.randn(n_samples + batch_size, self.dim)

        # Get the condition of sampling conditional
        condition = self.get_condition_for_sample(n_samples=n_samples, prior_samples=prior_samples,
                                                  batch_size=batch_size, con_depth=con_depth)


        # Wrapper function for the network
        # TODO: Replace with torchdiffeq
        def f(t, x_t, c=None):
            x_t_torch = torch.Tensor(x_t).reshape((batch_size, self.dim)).to(self.device)
            t_torch = t * torch.ones_like(x_t_torch[:, [0]])

            with torch.no_grad():
                if c is not None:
                    c_torch = torch.Tensor(c).reshape((batch_size, self.n_con)).to(self.device)
                    f_t = self.net(x_t_torch, t_torch, c_torch).detach().cpu().numpy().flatten()
                else:
                    f_t = self.net(x_t_torch, t_torch).detach().cpu().numpy().flatten()
            return f_t

        # Loop over batches and generate samples
        events = []
        with torch.no_grad():
            for i in range(int(n_samples / batch_size) + 1):
                # Get the condition
                if self.conditional:
                    c = condition[batch_size * i: batch_size * (i + 1)].flatten()
                else:
                    c = None
                # Use scipy.solve_ivp to generate samples
                sol = solve_ivp(f, (0, 1), x_T[batch_size * i: batch_size * (i + 1)].flatten(), args=[c])

                # Extract the solution and concat with condition if conditional
                if self.conditional:
                    c = condition[batch_size * i: batch_size * (i + 1)]
                    if self.n_jets == 1:
                        s = np.concatenate([c, sol.y[:, -1].reshape(batch_size, self.dim)], axis=1)
                    elif self.n_jets == 2:
                        s = np.concatenate([c[:,-2:], sol.y[:, -1].reshape(batch_size, self.dim)], axis=1)
                    elif self.n_jets == 3:
                        s = sol.y[:, -1].reshape(batch_size, self.dim)
                else:
                    s = sol.y[:, -1].reshape(batch_size, self.dim)

                events.append(s)

        return np.concatenate(events, axis=0)[:n_samples]

    def get_condition_for_sample(self, n_samples, batch_size, prior_samples, con_depth):
        """
        :param n_samples: number of samples
        :param batch_size: batch size of sampling process
        :param prior_samples: if not none, the samples produced with prior_model
        :param con_depth: number of models used priorly for the first dimensions
        :return: conditions as additional input for model during sampling
        """

        if self.conditional:

            if self.n_jets == 1 and con_depth == 0:
                n_c = (n_samples + batch_size) // 3
                n_r = (n_samples + batch_size) - 2 * n_c

                c_1 = np.array([[1, 0, 0]] * n_c)
                c_2 = np.array([[0, 1, 0]] * n_c)
                c_3 = np.array([[0, 0, 1]] * n_r)

                condition = np.concatenate([c_1, c_2, c_3])

            elif self.n_jets == 1 and con_depth == 1:
                n_c = (n_samples + batch_size) // 2
                n_r = (n_samples + batch_size) - n_c

                c_1 = np.array([[0, 1, 0]] * n_c)
                c_2 = np.array([[0, 0, 1]] * n_r)

                condition = np.concatenate([c_1, c_2])

            elif self.n_jets == 1 and con_depth == 2:
                n_c = n_samples + batch_size

                condition = np.array([[0, 0, 1]] * n_c)

            elif self.n_jets == 2:

                condition_1 = prior_samples[:, 3:12]
                condition_2 = prior_samples[:, 1:3]

                condition = np.concatenate([condition_1, condition_2], axis=1)

            else:
                condition = prior_samples[:, :13]

        else:
            condition = None

        return condition



    def invert_n(self, samples):
        """
        Start from samples and solve the reverse time ODE to get the according latent space samples
        """
        if self.net.bayesian:
            self.net.map = get(self.params,"fix_mu", False)
            for bay_layer in self.net.bayesian_layers:
                bay_layer.random = None
        self.eval()
        batch_size = get(self.params, "batch_size", 8192)
        n_samples = samples.shape[0]

        def f(t, x_t):
            x_t_torch = torch.Tensor(x_t).reshape((-1, self.dim)).to(self.device)
            t_torch = t * torch.ones_like(x_t_torch[:, [0]])
            with torch.no_grad():
                f_t = self.net(x_t_torch, t_torch).detach().cpu().numpy().flatten()
            return f_t

        events = []
        with torch.no_grad():
            for i in range(int(n_samples / batch_size)):
                sol = solve_ivp(f, (1, 0), samples[batch_size * i: batch_size * (i + 1)].flatten())
                s = sol.y[:, -1].reshape(batch_size, self.dim)
                events.append(s)
            sol = solve_ivp(f, (1, 0), samples[batch_size * (i+1):].flatten())
            s = sol.y[:, -1].reshape(-1, self.dim)
            events.append(s)
        return np.concatenate(events, axis=0)[:n_samples]

    # Generate samples and keep the complete diffusion trajectory
    def sample_n_evolution(self, n_samples):

        n_frames = get(self.params, "n_frames", 1000)
        t_frames = np.linspace(0, 1, n_frames)

        batch_size = get(self.params, "batch_size", 8192)
        x_T = np.random.randn(n_samples + batch_size, self.dim)

        def f(t, x_t):
            x_t_torch = torch.Tensor(x_t).reshape((batch_size, self.dim)).to(self.device)
            t_torch = t * torch.ones_like(x_t_torch[:, [0]])
            with torch.no_grad():
                f_t = self.net(x_t_torch, t_torch).detach().cpu().numpy().flatten()
            return f_t

        events = []
        with torch.no_grad():
            for i in range(int(n_samples / batch_size) + 1):
                sol = solve_ivp(f, (0, 1), x_T[batch_size * i: batch_size * (i + 1)].flatten(), t_eval=t_frames)
                s = sol.y.reshape(batch_size, self.dim, -1)
                events.append(s)
        return np.concatenate(events, axis=0)[:n_samples]



def sine_cosine_trajectory(x_0, x_1, t):
    c = torch.cos(t * np.pi / 2)
    s = torch.sin(t * np.pi / 2)
    x_t = c * x_0 + s * x_1

    c_dot = -np.pi / 2 * s
    s_dot = np.pi / 2 * c
    x_t_dot = c_dot * x_0 + s_dot * x_1
    return x_t, x_t_dot

def sine2_cosine2_trajectory(x_0, x_1, t):
    c = torch.cos(t * np.pi / 2)
    s = torch.sin(t * np.pi / 2)
    x_t = c**2 * x_0 + s**2 * x_1

    c_dot = -np.pi / 2 * s
    s_dot = np.pi / 2 * c
    x_t_dot = 2 * c_dot * c * x_0 + 2 * s_dot * s * x_1
    return x_t, x_t_dot

def linear_trajectory(x_0, x_1, t):
    x_t = (1 - t) * x_0 + t * x_1
    x_t_dot = x_1 - x_0
    return x_t, x_t_dot

def vp_trajectory(x_0, x_1, t, a=19.9, b=0.1):

    e = -1./4. * a * (1-t)**2 - 1./2. * b * (1-t)
    alpha_t = torch.exp(e)
    beta_t = torch.sqrt(1-alpha_t**2)
    x_t = x_0 * alpha_t + x_1 * beta_t

    e_dot = 2 * a * (1-t) + 1./2. * b
    alpha_t_dot = e_dot * alpha_t
    beta_t_dot = -2 * alpha_t * alpha_t_dot / beta_t
    x_t_dot = x_0 * alpha_t_dot + x_1 * beta_t_dot
    return x_t, x_t_dot


