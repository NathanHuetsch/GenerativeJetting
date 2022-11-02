import torch.optim
import torch.utils.data
import yaml


def load_params(argv):
    """Load the param file (first argument on program call)"""
    if isinstance(argv, str):
        with open(argv) as f:
            param = yaml.load(f, Loader=yaml.FullLoader)
            return param
    else:
        paramfile = argv[1]
        with open (paramfile) as f:
            param = yaml.load(f, Loader = yaml.FullLoader)
            return param


def save_params(params, name="paramfile.yaml"):
    with open(name, 'w') as f:
        yaml.dump(params, f)


def get_device():
    """Check whether cuda can be used"""
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    return device


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)


def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def get(dict, key, default):
    """
    small extension on dict.get()
    return value if exists, else return default and create entry
    """

    if key in dict:
        return dict[key]
    else:
        dict[key] = default
        return default
