import Source.Networks
from Source.Util.util import get
from Source.Models.ModelBase import GenerativeModel
import numpy as np
import torch
import time

class CM(GenerativeModel):
    def __init__(self, params):
        super().__init__(params)
        self.steps = get(self.params, "sample_steps", 1)
        
    def build_net(self):
        network = get(self.params, "network", "MLP")
        try:
            return getattr(Source.Networks, network)(self.params).to(self.device)
        except AttributeError:
            raise NotImplementedError(f"build_model: Network class {network} not recognised")

    def batch_loss(self, x, parent_model):
        '''
        Calculate batch_loss for CM
        t=0 -> x(0) noise t=1 -x(1)data
        '''
        with torch.no_grad():
            # Move these operations to GPU
            epsilon = torch.randn_like(x)
            t = torch.rand(x.shape[0], 1, device=x.device)
            x_t = t * x + (1 - t) * epsilon
            
            # Run parent model on GPU if possible
            v_theta = parent_model.net(x_t, t)
        
        # Euler Step
        stepsize = 0.01
        x_t_step = x_t + stepsize * v_theta
        t_step = t + stepsize

        f_theta, f_theta2 = self.forward_both(x_t, t, x_t_step, t_step)
        
        loss = torch.nn.functional.mse_loss(f_theta, f_theta2, reduction='mean')
        
        return loss

    def forward_both(self, x_t, t, x_t_step, t_step):
        return self.forward(x_t, t), self.forward(x_t_step, t_step)


    def forward(self, x, t):
        """Res. Net mit Formel aus Apendix B paper"""
        t = 1-t
        sigma = torch.tensor(0.5, dtype=x.dtype, device=x.device)
        epsilon = torch.tensor(1e-4, dtype=x.dtype, device=x.device)
        
        c_skip = sigma ** 2 / ((t - epsilon) ** 2 + sigma ** 2)
        c_out = sigma * (t - epsilon) / torch.sqrt(sigma ** 2 + t ** 2)
                
        return c_skip * x + c_out * self.net(x, t)  ##Forward statt 
    

    def sample_n(self, nsamples:int, steps = None):
        """
        Sample Data in N steps
        from t = 1 and x(1) = noise to t = 0 and x(0) = noise
        """
        if steps is None: steps = self.steps
        
        start_time = time.time()
        epsilon = torch.randn(nsamples, self.dim_x, device=self.device)
        batch_size = get(self.params, "batch_size_sample", 8192)
        
        batches = torch.split(epsilon, batch_size)
        events = []
        calls = []
        with torch.no_grad():
            self.eval2 = 0 
            for batch in batches:
                t = torch.zeros(batch.shape[0], 1, device = self.device).float()
                x = self.forward(batch, t) #Hier self.net oder forward?? 
                self.eval2 = self.eval2 + 1
                if steps > 1:
                    for s in range(steps-1):
                        z = torch.randn(batch.shape[0], self.dim_x, device = self.device)
                        t += 1/steps 
                        x = t * x + (1-t) * z
                        x = self.forward(x, t)
                        self.eval2 = self.eval2 + 1
                    
                x = x.to('cpu')
                calls.append(self.eval2)
                events.append(x)
                self.eval2 = 0
            stop_time = time.time()
            print(f"generate_samples: Finished generation of {nsamples} samples with {np.mean(calls)} steps after {(stop_time-start_time):.2f}s ")
            return np.concatenate(events)       


