import Source.Networks
from Source.Util.util import get
from Source.Models.ModelBase import GenerativeModel
import numpy as np
import torch

class CM(GenerativeModel):
#MUST OVERWRITE 
    #def build_net(self): should register some NN architecture as self.net
    #def batch_loss(self, x): takes a batch of samples as input and returns the loss
    #def sample_n_parallel(self, n_samples): generates and returns n_samples new samples
    def __init__(self, params):
        super().__init__(params)

    def build_net(self):
        network = get(self.params, "network", "MLP")
        try:
            return getattr(Source.Networks, network)(self.params).to(self.device)
        except AttributeError:
            raise NotImplementedError(f"build_model: Network class {network} not recognised")


    def batch_loss(self, x, parent_model):
        '''cal. batch_loss for CM '''
        stepsize = 0.01       

        # Gen random punkt zwischen noise und data
        epsilon = torch.randn_like(x, device = x.device)
        t = torch.rand(x.shape[0], 1, device= x.device)
        x_t = (1 - t) * x + t * epsilon

        # Velocity
        v_theta = parent_model.net(x_t, t).detach()

        # Euler Step
        x_t_step = x_t + stepsize * v_theta
        t_step = t + stepsize

        # Consitency model forward
        f_theta = self.forward(x_t, t)
        f_theta2 = self.forward(x_t_step, t_step)

        loss = torch.mean((f_theta - f_theta2) ** 2)
        return loss
    

    def forward(self, x0, t):
        """Res. Net mit Formel aus Apendix B paper"""

        sigma = torch.tensor(0.5, dtype=x0.dtype, device=x0.device)
        epsilon = torch.tensor(1e-4, dtype=x0.dtype, device=x0.device)
        
        c_skip = sigma ** 2 / ((t - epsilon) ** 2 + sigma ** 2)
        c_out = sigma * (t - epsilon) / torch.sqrt(sigma ** 2 + t ** 2)

        return c_skip * x0 + c_out * self.net(x0, t)  ##Forward statt 
    

    def sample_n(self, nsamples, steps=None):
        if steps is None: steps = 1

        """
        Sample Data in N steps
        from t = 1 and x(1) = noise to t = 0 and x(0) = noise
        """
        epsilon = torch.randn(nsamples, self.dim_x, device=self.device)
        batch_size = 1000
        batches = torch.split(epsilon, batch_size)
        events = []
        with torch.no_grad():
            for batch in batches:
                t = torch.ones(batch.shape[0], 1, device = self.device).float()
                x = self.forward(batch, t)
                for _ in range(steps):
                    z = torch.randn(batch.shape[0], self.dim_x, device = self.device)
                    t -= 1/steps
                    x = (1-t) * x + t * z
                    x = self.forward(x, t)
                x = x.to('cpu')
                events.append(x)

            print(f"generate_samples: Finished generation of {nsamples} samples with {steps}  s")
            return np.concatenate(events)       


