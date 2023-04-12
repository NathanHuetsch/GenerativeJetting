import torch
import torch.nn as nn

class EnergyNet(nn.Module):
    """
    Simple multilayer perceptron used to model the pdf in the autoregNN framework
    Called EnergyNet, because it predicts only the energy (without normalization constant), see energy-based models
    """
    def __init__(self, param, out=True):
        super().__init__()
        # Read in the network specifications from the params
        self.param = param

        self.n_conditions = param["n_conditions"]
        self.layer_size = param.get("enet_layer_size", [round(2/3*self.n_conditions), round(1/3*self.n_conditions)])
        self.n_layers = len(self.layer_size)
        self.bias = param.get("enet_bias", True)
        self.activation = param.get("enet_activation", "ReLU")

        prec_mean = param.get("enet_prec_mean", param["intermediate_dim"])
        if out:
            print(f"energynet hyperparameters: enet_layer_size={self.layer_size}, enet_bias={self.bias}, "\
              f"enet_activation={self.activation}, n_conditions={self.n_conditions}, enet_prec_mean={prec_mean}")
        # Build the model
        layers = []
        layers.append(nn.Linear(self.n_conditions+1, self.layer_size[0], bias=self.bias))
        layers.append(getattr(nn, self.activation)())
        for i in range(1, self.n_layers):
            layers.append(nn.Linear(self.layer_size[i-1], self.layer_size[i], bias=self.bias))
            layers.append(getattr(nn, self.activation)())
        layers.append(nn.Linear(self.layer_size[-1], 1, bias=self.bias))

        self.layers = nn.ModuleList(layers)
        self.model_parameters = sum(p.numel() for p in self.parameters())
        
    def forward(self, x, conditions):
        x = x.unsqueeze(3)
        conditions = conditions.reshape(conditions.size(0), conditions.size(1), 1, conditions.size(2)) #generate 2nd component (different x values)
        conditions = conditions.expand(conditions.size(0), conditions.size(1), x.size(2), conditions.size(3)) #same conditions for each x value
        y = torch.cat((x, conditions), axis=-1)
        for layer in self.layers:
            y = layer(y)
        y = y.reshape(y.size(0), y.size(1), y.size(2))
        return y
