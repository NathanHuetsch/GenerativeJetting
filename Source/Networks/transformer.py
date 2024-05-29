import torch
import torch.nn as nn
from Source.Networks.vblinear import VBLinear
import math

class Transformer(nn.Module):

    def __init__(self, param):
        super().__init__()
        # Read in the network specifications from the params
        self.params = param

        self.dim_embedding = self.params["dim_embedding"]
        self.dims_in = self.params["dim_x"]
        self.transformer = nn.Transformer(
            d_model=self.dim_embedding,
            nhead=self.params["n_head"],
            num_encoder_layers=self.params["n_encoder_layers"],
            num_decoder_layers=self.params["n_decoder_layers"],
            dim_feedforward=self.params["dim_feedforward"],
            dropout=self.params.get("dropout", 0.0),
            activation=self.params.get("activation", "relu"),
            batch_first=True,
        )
        self.bayesian = False

        self.embeds = self.params.get("embeds", False)
        if self.embeds:
            print("Using embeds")
            #self.x_embed = SinCos_embedding(n_frequencies=self.dim_embedding/2)
            self.x_embed = nn.Linear(1, int(self.dim_embedding/2))
            #self.pos_embed = PositionalEncoding(d_model=self.dim_embedding)
            self.pos_embed = nn.Embedding(self.dims_in, int(self.dim_embedding/2))
            self.time_embed = nn.Linear(1, int(self.dim_embedding/2))
            self.layer = nn.Linear(self.dim_embedding + int(self.dim_embedding/2), 1)
        else:
            self.layer = nn.Linear(self.dim_embedding + 1, 1)



    def compute_embedding(
        self, p: torch.Tensor, n_components: int, t: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Appends the one-hot encoded position to the momenta p. Then this is either zero-padded
        or an embedding net is used to compute the embedding of the correct dimension.
        """
        if self.embeds:
            p = self.x_embed(p.unsqueeze(-1))
            p = p + self.pos_embed(torch.arange(n_components, device=p.device))
            if t is not None:
                t = self.time_embed(t).unsqueeze(1)
                return torch.cat([t.repeat(1, p.size(1), 1), p], dim=-1)
            else:
                return p
        else:
            one_hot = torch.eye(n_components, device=p.device, dtype=p.dtype)[
                None, : p.shape[1], :
            ].expand(p.shape[0], -1, -1)
            if t is None:
                p = p.unsqueeze(-1)
            else:
                p = torch.cat([p.unsqueeze(-1), t.unsqueeze(-1).expand(t.shape[0], p.shape[1], 1)], dim=-1)
            n_rest = self.dim_embedding - n_components - p.shape[-1]
            assert n_rest >= 0
            zeros = torch.zeros((*p.shape[:2], n_rest), device=p.device, dtype=p.dtype)
            return torch.cat((p, one_hot, zeros), dim=-1)

    def forward(self, x, t, condition=None):
        """
        forward method of our Resnet
        """
        self.kl = 0

        if condition is None:
            embedding = self.transformer.decoder(
                tgt=self.compute_embedding(
                    x,
                    n_components=self.dims_in,
                    t=t
                ),
                memory=torch.zeros((x.size(0), x.size(1), self.dim_embedding), device=x.device, dtype=x.dtype)
            )
        else:
            embedding = self.transformer(
                src=self.compute_embedding(
                    condition,
                    n_components=self.dims_c
                ),
                tgt=self.compute_embedding(
                    x,
                    n_components=self.dims_in,
                    t=t
                )
            )

        if self.embeds:
            t = self.time_embed(t)

        v_pred = self.layer(torch.cat([t.unsqueeze(1).repeat(1, x.size(1), 1), embedding], dim=-1)).squeeze()
        return v_pred


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SinCos_embedding(nn.Module):

    def __init__(self, n_frequencies: int, sigmoid=True):
        super().__init__()
        self.arg = nn.Parameter(2 * math.pi * 2**torch.arange(n_frequencies), requires_grad=False)
        self.sigmoid = sigmoid

    def forward(self, x):
        if self.sigmoid:
            x_pp = nn.functional.sigmoid(x)
        else:
            x_pp = x
        frequencies = (x_pp.unsqueeze(-1)*self.arg).reshape(x_pp.size(0), x_pp.size(1), -1)
        return torch.cat([torch.sin(frequencies), torch.cos(frequencies)], dim=-1)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding