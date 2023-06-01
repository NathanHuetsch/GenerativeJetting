# Basic python libraries
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

# Other classes in project
from Source.Util.util import linear_beta_schedule, cosine_beta_schedule, get
from Source.Models.ModelBase import GenerativeModel
import Source.Networks

class DDPM(GenerativeModel):

    def __init__(self, params):
        """
        :param params: need file with all model specific parameters, if empty default values will be used
        """
        super().__init__(params)

        # get DDPM-specific hyperparameters
        # get number of diffusive timesteps
        self.timesteps = get(self.params, "timesteps", 1000)

        # get functional form of the step-wise variance of the forward diffusion process
        self.beta_schedule = get(self.params, "beta_schedule", "linear")

        # get actual values of the step-wise variance of the forward diffusion process depending on the functional form
        if self.beta_schedule == 'linear':
            self.register_buffer('betas', linear_beta_schedule(self.timesteps))
        elif self.beta_schedule == 'cosine':
            self.register_buffer('betas', cosine_beta_schedule(self.timesteps))
        else:
            raise ValueError(f'unknown beta schedule {self.beta_schedule}')

        # calculate and save some coefficients (Notation from Ho et. al. 2020) all dependent on beta
        self.register_buffer('alphas', 1-self.betas)
        self.register_buffer('alphas_bar', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_bar_prev', F.pad(self.alphas_bar[:-1], (1, 0), value=1.))
        self.register_buffer('One_minus_alphas_bar', 1-self.alphas_bar)
        self.register_buffer('sqrt_alphas', torch.sqrt(self.alphas))
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(self.alphas_bar))
        self.register_buffer('sqrt_One_minus_alphas_bar', torch.sqrt(self.One_minus_alphas_bar))
        self.register_buffer('betas_tilde', self.betas * (1-self.alphas_bar_prev) / (1-self.alphas_bar))

        # define standard deviation of the reverse diffusion process, either sigma^2 = beta or sigma^2 = beta_tilde
        self.sigma_mode = get(self.params, "sigma_mode", "beta")
        if self.sigma_mode == "beta":
            self.register_buffer("sigmas", torch.sqrt(self.betas))
        elif self.sigma_mode == "beta_tilde":
            self.register_buffer("sigmas", torch.sqrt(self.betas_tilde))
        else:
            raise ValueError(f'unknown sigma mode {self.sigma_mode}')

        # set prefactor in front of bayesian loss term if necessary
        self.C = get(self.params, "C", 1)
        if self.C != 1:
            print(f"C is {self.C}")

        # check if loss should be weighted according to the magic transformation
        self.magic_transformation = get(self.params, "magic_transformation", False)

        self.to(self.device)
    def build_net(self):
        """
        Build the network defined in Networks
        """
        network = get(self.params, "network", "Resnet")
        try:
            return getattr(Source.Networks, network)(self.params).to(self.device)
        except AttributeError:
            raise NotImplementedError(f"build_model: Network class {network} not recognised")

    def get_relative_factor(self,t):
        """
        :param t: timestep
        :return: time-dependent prefactor for loss as described in Ho et. al. 2020
        """
        return self.betas[t]/(np.sqrt(2) * self.sigmas[t] * self.sqrt_alphas[t]*self.sqrt_One_minus_alphas_bar[t])
    def xT_from_x0_and_noise(self, x0, t, noise):
        # Forward process
        """
        :param x0: point in phase space
        :param t: timestep
        :param noise: Gaussian noise
        :return: diffused point x_t
        """
        return self.sqrt_alphas_bar[t]*x0 + self.sqrt_One_minus_alphas_bar[t]*noise

    def x0_from_xT_and_noise(self, xT, t, noise):
        # Forward process
        """
        :param xT: diffused point x_t
        :param t: timestep
        :param noise: Gaussian noise
        :return: phase space point
        """
        return (1./self.sqrt_alphas_bar[t]) * (xT - self.sqrt_One_minus_alphas_bar[t] * noise)

    def mu_tilde_t(self, xT, t, noise):
        """
        :param xT: diffused point x_t
        :param t: timestep
        :param noise: Gaussian noise
        :return: mean of convoluted Gaussian at time t
        """
        return (1./self.sqrt_alphas[t]) * (xT - noise*self.betas[t]/self.sqrt_One_minus_alphas_bar[t])
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
    def batch_loss(self, x):
        """
        :param x: input
        :return: loss
        """

        # set all bayesian weights to none deterministic ones
        self.net.map = False

        # get input and conditions
        x, condition, weights = self.get_condition_and_input(x)

        T = self.timesteps
        # sample ts
        t = torch.randint(low=1, high=T, size=(x.size(0), 1), device=self.device)

        # sample Gaussian noise
        noise = torch.randn_like(x, device=self.device)

        # calculate diffused point x_t
        xT = self.xT_from_x0_and_noise(x, t, noise)

        # feed diffused point x_t, timestep t and the condition to model and get prediction for epsilon_theta
        model_pred = self.net(xT.float(), t, condition)

        # get prefactor of loss
        c = self.get_relative_factor(t)

        # if true compute weighted loss
        if self.magic_transformation:
            loss = torch.mean(c*(model_pred - noise) ** 2 * weights[:, None])/weights.mean() \
                   + self.C*self.net.kl / (len(self.data_train)*T)
            # save loss for loss plot
            self.regular_loss.append((torch.mean(c*(model_pred - noise) ** 2 * weights[:, None])/weights.mean()).detach().cpu().numpy())
        else:
            loss = torch.mean(c*(model_pred - noise)**2) + self.C*self.net.kl / (len(self.data_train)*T)
            # save loss for loss plot
            self.regular_loss.append(F.mse_loss(c*model_pred, c*noise).detach().cpu().numpy())
        try:
            # save bayesian loss for loss plot
            self.kl_loss.append((self.C*self.net.kl / (len(self.data_train)*T)).detach().cpu().numpy())
        except:
            pass

        return loss

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

    def sample_n(self, n_samples, prior_samples=None, con_depth=0):
        """
        :param n_samples: Number of samples
        :param prior_samples: if not none, the samples produced with prior_model
        :param con_depth: number of models used priorly for the first dimensions
        :return: numpy_darray of newly generated samples in phase space
        """
        if self.net.bayesian:
            # if true, weights are not drawn but fixed to mean
            self.net.map = get(self.params,"fix_mu", False)
            # make sure that for each sample iteration new weights are sampled
            for bay_layer in self.net.bayesian_layers:
                bay_layer.random = None

        self.eval()
        batch_size = self.batch_size_sample
        events = []

        # get conditions for different networks
        condition = self.get_condition_for_sample(n_samples=n_samples, prior_samples=prior_samples,
                                                  batch_size=batch_size, con_depth=con_depth)

        for i in range(int(n_samples / batch_size) + 1):
            if self.conditional:
                # reshape condition
                c = torch.from_numpy(condition[batch_size * i: batch_size * (i + 1)]).to(self.device)
            else:
                c = None
            # draw random noise z_t (as in paper)
            noise_i = torch.randn(self.timesteps, batch_size, self.dim).to(self.device)
            # define x_T point in latent space
            x = noise_i[0]
            # do the reverse diffusion process from T -> 0
            for t in reversed(range(self.timesteps)):
                z = noise_i[t] if t > 0 else 0
                with torch.no_grad():
                    # get model prediction epsilon_theta from input, condition and timestep t
                    model_pred = self.net(x, t*torch.ones_like(x[:, [0]],dtype=torch.int), c).detach()
                # draw x_t-1 from p(x_t-1 | x_t)
                x = self.mu_tilde_t(x, t, model_pred) + self.sigmas[t] * z

            # concatenate samples + condition if necessary
            if self.conditional and self.n_jets == 1:
                s = torch.concatenate([c,x], dim=1)
            elif self.conditional and self.n_jets == 2:
                s = torch.concatenate([c[:, -2:],x], dim=1)
            else:
                s = x
            events.append(s.cpu().numpy())
        return np.concatenate(events, axis=0)[:n_samples]

    def sample_joint_forward(self, x0, n):
        """
        :param x0: phase space point
        :param n: number of samples
        :return: x_0 + sampled (x_1,...,x_T) and the log probability of the joint forward process p(x_1,...,x_T | x_0)
        """

        log_q = torch.zeros(n)
        xs = torch.zeros(self.timesteps + 1, n, self.dim)

        xs[0] = x0
        for t in range(self.timesteps):
            variance = self.betas[t]
            mean = self.sqrt_alphas[t] * xs[t]
            x = torch.normal(mean, torch.sqrt(variance))
            q = MultivariateNormal(loc=mean, covariance_matrix=variance * torch.eye(self.dim)).log_prob(x)
            xs[t + 1] = x
            log_q += q

        return xs, log_q

    def get_joint_log_p(self, xn):
        """
        :param xn: n samples of (x_0,...,x_T)
        :return: log probability of the learned reverse diffusion process p((x_0,...,x_T)| theta)
        """

        n = xn.shape[1]
        log_q = torch.zeros(n)

        log_q += MultivariateNormal(loc=torch.zeros(self.dim), covariance_matrix=torch.eye(self.dim)).log_prob(
            xn[self.timesteps])

        for t in range(0, self.timesteps):
            with torch.no_grad():
                model_pred = self.net(xn[t + 1], t * torch.ones_like(xn[t][:, [0]], dtype=torch.int)).detach()
            mean = self.mu_tilde_t(xn[t + 1], t, model_pred)
            variance = self.sigmas[t] ** 2 * torch.eye(self.dim)
            p = MultivariateNormal(loc=mean, covariance_matrix=variance).log_prob(xn[t])
            log_q += p

        return log_q


    def get_likelihood(self,x0,n):
        #Calculate modeled likelihood by imporantant sampling as described in the paper
        """
        :param x0: phase space point
        :param n: number of samples
        :return: log likelihood of model log(p_model(x_0 | theta))
        """
        #get n samples (x_1,...,x_n) and the corresponding forward log probability p((x_1,...,x_T)|x_0)
        x, log_q = self.sample_joint_forward(x0,n)

        #get modeled joint likelihood of the entire diffusion process p((x_0,...,x_T)|theta)
        log_p = self.get_joint_log_p(x)

        # calculate the log likelihood ratio of joint forward log likelihood and modeled likelihood
        l = log_p - log_q

        #  calculate the log mean likelihood from our n samples
        p = torch.logsumexp(l) - torch.log(n)

        return p

