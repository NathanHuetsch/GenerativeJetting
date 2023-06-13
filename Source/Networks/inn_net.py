import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from FrEIA.framework import *
from FrEIA.modules import *

from typing import Callable
from scipy.stats import special_ortho_group
import numpy as np
import math

from Source.Networks.vblinear import VBLinear

"""
The necessary code to build the INNs.
Any questions are best asked to Theo. I have no clue.
"""
class INNnet(nn.Module):
    def __init__(self, params):
        super().__init__()
        # Load all hyperparameters
        self.params = params

        #Get bayesian parameters
        self.bayesian = self.params.get("bayesian", False)
        self.prior_prec = self.params.get("prior_prec", 1.)

        # set input and conditional dimension
        self.dim = self.params.get("dim", 4)
        self.conditional = self.params.get("conditional",False)
        self.n_con = self.params.get("n_con", 3)

        # set network parameters
        self.intermediate_dim = self.params.get("intermediate_dim", 128)
        self.layers_per_block = self.params.get("layers_per_block", 4)
        self.n_blocks = self.params.get("n_blocks", 4)

        # set spline parameters
        self.use_splines = self.params.get("use_splines", False)
        self.permutation_seed = self.params.get("seed", None)
        self.clamping = self.params.get("clamping", 2.)
        self.spline_bound = self.params.get("spline_bound", 1.)

        # special layers
        self.dropout = self.params.get("dropout", 0.0)

        # build subnet function
        self.constructor_fct = partial(create_fully_connected_net, dims_intern=self.intermediate_dim,
                                       num_layers=self.layers_per_block, dropout=self.dropout, bayesian=self.bayesian,
                                       prior_prec= self.prior_prec)
        # build INN
        self.inn = self.build()

        # build list of bayesian layers to keep track of kl term
        if self.bayesian:
            self.bayesian_modules = []
            for module in self.modules():
                if hasattr(module, "KL") and module != self:
                    self.bayesian_modules.append(module)

    def get_coupling_block(self):
        # Define subnet construction function
        block_kwargs = {
            "subnet_constructor": self.constructor_fct }

        # If true, build cubic spline blocks and set needed parameters
        if self.use_splines == "cubic":
            CouplingBlock = CubicSplineBlock
            block_kwargs['num_bins'] = self.params.get("num_bins", 10)
            block_kwargs['bounds_init'] = self.spline_bound
            block_kwargs['permute_soft'] = self.params.get("permute_soft", True)

        # If true, build quadratic spline blocks and set needed parameters
        elif self.use_splines == "quadratic":
            CouplingBlock = QuadraticSplineBlock
            block_kwargs['num_bins'] = self.params.get("num_bins", 10)
            block_kwargs['bounds_init'] = self.spline_bound
            block_kwargs['permute_soft'] = self.params.get("permute_soft", True)

        # Else build simple affine coupling blocks
        else:
            CouplingBlock = AllInOneBlock
            block_kwargs['affine_clamping'] = self.clamping

        return CouplingBlock, block_kwargs

    def build(self):
        # Define list of nodes
        nodes = []

        # if true, build conditional node and append to nodes
        if self.conditional:
            node_condition = ConditionNode(self.n_con)
            nodes.append(node_condition)
        else:
            node_condition = []

        # Define input node
        nodes.append(InputNode(self.dim,1, name='input'))
        nodes.append(Node(nodes[-1], Flatten, {}, name='flatten'))

        # Define coupling blocks and needed arguments
        CouplingBlock, block_kwargs = self.get_coupling_block()

        # Create n additional intermediate nodes where the condition is fed into
        for i in range(self.params.get("n_blocks", 10)):
            nodes.append(Node(nodes[-1], CouplingBlock, block_kwargs, name=f"block_{i}", conditions=node_condition))

        # Create output node
        nodes.append(OutputNode(nodes[-1], name='out'))

        # build INN out of list of nodes
        model = GraphINN(nodes, verbose=False)

        return model

    def reset_weight_samples(self):
        assert hasattr(self, "bayesian_modules")

        # make sure that for each sample iteration new weights are sampled
        for module in self.bayesian_modules:
            module.random = None

    def eval(self):
        super().eval()
        # reset weights for each bayesian layer
        if self.bayesian:
            self.reset_weight_samples()

    def kl(self):
        # if true, sums over the kl terms of all bayesian layers
        if hasattr(self, "bayesian_modules"):
            kl = sum(module.KL() for module in self.bayesian_modules)
        else:
            kl = 0
        return kl

    def forward(self, x, c=[], rev=False, jac=True):
        x_out, log_jacobian_det =self.inn(x,c,rev=rev, jac=jac)

        if rev:
            return x_out[...,0], log_jacobian_det

        return x_out, log_jacobian_det


class CubicSplineBlock(InvertibleModule):

    # default parameters for bin properties
    DEFAULT_MIN_BIN_WIDTH = 1e-3
    DEFAULT_MIN_BIN_HEIGHT = 1e-3
    DEFAULT_EPS = 1e-5
    DEFAULT_QUADRATIC_THRESHOLD = 1e-3

    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor: Callable = None,
                 num_bins: int = 10,
                 bounds_init: float = 1.,
                 permute_soft: bool = False,
                 tails='linear',
                 bounds_type="SOFTPLUS"):

        super().__init__(dims_in, dims_c)
        # specify input dimension
        channels = dims_in[0][0]
        # rank of the tensors means 1d, 2d, 3d tensor etc.
        self.input_rank = len(dims_in[0]) - 1
        # tuple containing all dims except for batch-dim (used at various points)
        self.sum_dims = tuple(range(1, 2 + self.input_rank))

        # check for conditional dimensions
        if len(dims_c) == 0:
            self.conditional = False
            self.condition_channels = 0
        else:
            assert tuple(dims_c[0][1:]) == tuple(dims_in[0][1:]), \
                F"Dimensions of input and condition don't agree: {dims_c} vs {dims_in}."
            self.conditional = True
            # set numbers (dimension) of conditions
            self.condition_channels = sum(dc[0] for dc in dims_c)

        # create two coupling blocks of equal length input_dimension/2 (+-1)
        split_len1 = channels - channels // 2
        split_len2 = channels // 2
        self.splits = [split_len1, split_len2]

        # set num_bins
        self.num_bins = num_bins

        if self.DEFAULT_MIN_BIN_WIDTH * self.num_bins > 1.0:
            raise ValueError('Minimal bin width too large for the number of bins')
        if self.DEFAULT_MIN_BIN_HEIGHT * self.num_bins > 1.0:
            raise ValueError('Minimal bin height too large for the number of bins')

        try:
            self.permute_function = {0: F.linear,
                                     1: F.conv1d,
                                     2: F.conv2d,
                                     3: F.conv3d}[self.input_rank]
        except KeyError:
            raise ValueError(f"Data is {1 + self.input_rank}D. Must be 1D-4D.")

        # set domain bounds depending on the bounds_type
        if bounds_type == 'SIGMOID':
            bounds = 2. - np.log(10. / bounds_init - 1.)
            self.bounds_activation = (lambda a: 10 * torch.sigmoid(a - 2.))
        elif bounds_type == 'SOFTPLUS':
            bounds = 2. * np.log(np.exp(0.5 * 10. * bounds_init) - 1)
            self.softplus = nn.Softplus(beta=0.5)
            self.bounds_activation = (lambda a: 0.1 * self.softplus(a))
        elif bounds_type == 'EXP':
            bounds = np.log(bounds_init)
            self.bounds_activation = (lambda a: torch.exp(a))
        else:
            raise ValueError('Global affine activation must be "SIGMOID", "SOFTPLUS" or "EXP"')

        self.in_channels = channels
        #self.bounds = nn.Parameter(torch.ones(1, self.splits[1], *([1] * self.input_rank)) * float(bounds))
        self.bounds =  self.bounds_activation(torch.ones(1, self.splits[1], *([1] * self.input_rank)) * float(bounds))
        self.tails = tails

        # permute input channels, either by rotation or randomly
        if permute_soft:
            w = special_ortho_group.rvs(channels)
        else:
            w = torch.zeros((channels, channels))
            for i, j in enumerate(np.random.permutation(channels)):
                w[i, channels-i-1] = 1.

        # apply permutation
        self.w_perm = nn.Parameter(torch.FloatTensor(w).view(channels, channels, *([1] * self.input_rank)),
                                   requires_grad=False)
        self.w_perm_inv = nn.Parameter(torch.FloatTensor(w.T).view(channels, channels, *([1] * self.input_rank)),
                                       requires_grad=False)

        # define subnet (conditioner)
        if subnet_constructor is None:
            raise ValueError("Please supply a callable subnet_constructor"
                             "function or object (see docstring)")
        self.subnet = subnet_constructor(self.splits[0] + self.condition_channels, (2 * self.num_bins + 2) * self.splits[1])
        self.last_jac = None

    def _unconstrained_cubic_spline(self,
                                   inputs,
                                   theta,
                                   rev=False):
        """
        :param inputs: untransformed coupling block
        :param theta: transformed coupling block (output of subnet) [:,bin_widths + bin_heights + left_derivative +
                      right_derivative)
        :param rev: if true, process is reversed (for sampling <-)
        :return: spline output (tau(input,theta)) and log|det(Jx)|
        """

        # get input values which are inside the spline domain
        inside_interval_mask = torch.all((inputs >= -self.bounds) & (inputs <= self.bounds),
                                         dim = -1)

        # get input values which are outside the spline domain
        outside_interval_mask = ~inside_interval_mask

        # define empty masks to store output and log|det(Jx)|
        masked_outputs = torch.zeros_like(inputs)
        masked_logabsdet = torch.zeros(inputs.shape[0], device=inputs.device)

        if self.tails == 'linear':
            # transform input values outside of spline domain with identy matrix, i.e. return input value as output
            masked_outputs[outside_interval_mask] = inputs[outside_interval_mask]
            # those input points don't contribute to the Jacobian
            masked_logabsdet[outside_interval_mask] = 0
        else:
            raise RuntimeError('{} tails are not implemented.'.format(self.tails))

        # define input to only contain points in spline domain
        inputs = inputs[inside_interval_mask]

        # define output from subnet to only contain points, where input is in spline domain
        theta = theta[inside_interval_mask, :]

        # set spline parameters
        min_bin_width=self.DEFAULT_MIN_BIN_WIDTH
        min_bin_height=self.DEFAULT_MIN_BIN_HEIGHT
        eps=self.DEFAULT_EPS
        quadratic_threshold=self.DEFAULT_QUADRATIC_THRESHOLD

        # set spline bound
        bound = torch.min(self.bounds)
        left = -bound
        right = bound
        bottom = -bound
        top = bound

        if not rev and (torch.min(inputs).item() < left or torch.max(inputs).item() > right):
            raise ValueError("Spline Block inputs are not within boundaries")
        elif rev and (torch.min(inputs).item() < bottom or torch.max(inputs).item() > top):
            raise ValueError("Spline Block inputs are not within boundaries")

        # get predicted bin_widths and bin_heights for each bin (num_bins) as well as the derivative on left and right
        # boundary
        unnormalized_widths = theta[...,:self.num_bins]
        unnormalized_heights = theta[...,self.num_bins:self.num_bins*2]
        unnorm_derivatives_left = theta[...,-2].reshape(theta.shape[0], self.splits[1], 1)
        unnorm_derivatives_right = theta[...,-1].reshape(theta.shape[0], self.splits[1], 1)

        # if true axis are interchanged for normalization of input
        if rev:
            inputs = (inputs - bottom) / (top - bottom)
        else:
            inputs = (inputs - left) / (right - left)

        # normalize bin widths
        widths = F.softmax(unnormalized_widths, dim=-1)
        widths = min_bin_width + (1 - min_bin_width * self.num_bins) * widths

        # get cumulated bin width of all bins in num_bins
        cumwidths = torch.cumsum(widths, dim=-1)
        cumwidths[..., -1] = 1
        cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)

        # normalize bin heights
        heights = F.softmax(unnormalized_heights, dim=-1)
        heights = min_bin_height + (1 - min_bin_height * self.num_bins) * heights

        # get cumulated bin height of all bins in num_bins
        cumheights = torch.cumsum(heights, dim=-1)
        cumheights[..., -1] = 1
        cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)

        # calculate slope in each bin
        slopes = heights / widths
        min_something_1 = torch.min(torch.abs(slopes[..., :-1]),
                                    torch.abs(slopes[..., 1:]))
        min_something_2 = (
                0.5 * (widths[..., 1:] * slopes[..., :-1] + widths[..., :-1] * slopes[..., 1:])
                / (widths[..., :-1] + widths[..., 1:])
        )
        min_something = torch.min(min_something_1, min_something_2)

        # normalize left and right derivative
        derivatives_left = torch.sigmoid(unnorm_derivatives_left) * 3 * slopes[..., 0][..., None]
        derivatives_right = torch.sigmoid(unnorm_derivatives_right) * 3 * slopes[..., -1][..., None]

        derivatives = min_something * (torch.sign(slopes[..., :-1]) + torch.sign(slopes[..., 1:]))
        derivatives = torch.cat([derivatives_left,
                                 derivatives,
                                 derivatives_right], dim=-1)
        a = (derivatives[..., :-1] + derivatives[..., 1:] - 2 * slopes) / widths.pow(2)
        b = (3 * slopes - 2 * derivatives[..., :-1] - derivatives[..., 1:]) / widths
        c = derivatives[..., :-1]
        d = cumheights[..., :-1]

        # if true, axis are interchanged
        if rev:
            bin_idx = self.searchsorted(cumheights, inputs)[..., None]
        else:
            bin_idx = self.searchsorted(cumwidths, inputs)[..., None]

        inputs_a = a.gather(-1, bin_idx)[..., 0]
        inputs_b = b.gather(-1, bin_idx)[..., 0]
        inputs_c = c.gather(-1, bin_idx)[..., 0]
        inputs_d = d.gather(-1, bin_idx)[..., 0]

        input_left_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
        input_right_cumwidths = cumwidths.gather(-1, bin_idx + 1)[..., 0]

        # invert spline functions bin-wise
        if rev:
            # Modified coefficients for solving the cubic.
            inputs_b_ = (inputs_b / inputs_a) / 3.
            inputs_c_ = (inputs_c / inputs_a) / 3.
            inputs_d_ = (inputs_d - inputs) / inputs_a

            delta_1 = -inputs_b_.pow(2) + inputs_c_
            delta_2 = -inputs_c_ * inputs_b_ + inputs_d_
            delta_3 = inputs_b_ * inputs_d_ - inputs_c_.pow(2)

            discriminant = 4. * delta_1 * delta_3 - delta_2.pow(2)

            depressed_1 = -2. * inputs_b_ * delta_1 + delta_2
            depressed_2 = delta_1

            three_roots_mask = discriminant >= 0  # Discriminant == 0 might be a problem in practice.
            one_root_mask = discriminant < 0

            outputs = torch.zeros_like(inputs)

            # Deal with one root cases.

            p = self.cbrt((-depressed_1[one_root_mask] + torch.sqrt(-discriminant[one_root_mask])) / 2.)
            q = self.cbrt((-depressed_1[one_root_mask] - torch.sqrt(-discriminant[one_root_mask])) / 2.)

            outputs[one_root_mask] = ((p + q)
                                      - inputs_b_[one_root_mask]
                                      + input_left_cumwidths[one_root_mask])

            # Deal with three root cases.

            theta = torch.atan2(torch.sqrt(discriminant[three_roots_mask]), -depressed_1[three_roots_mask])
            theta /= 3.

            cubic_root_1 = torch.cos(theta)
            cubic_root_2 = torch.sin(theta)

            root_1 = cubic_root_1
            root_2 = -0.5 * cubic_root_1 - 0.5 * math.sqrt(3) * cubic_root_2
            root_3 = -0.5 * cubic_root_1 + 0.5 * math.sqrt(3) * cubic_root_2

            root_scale = 2 * torch.sqrt(-depressed_2[three_roots_mask])
            root_shift = (-inputs_b_[three_roots_mask] + input_left_cumwidths[three_roots_mask])

            root_1 = root_1 * root_scale + root_shift
            root_2 = root_2 * root_scale + root_shift
            root_3 = root_3 * root_scale + root_shift

            root1_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_1).float()
            root1_mask *= (root_1 < (input_right_cumwidths[three_roots_mask] + eps)).float()

            root2_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_2).float()
            root2_mask *= (root_2 < (input_right_cumwidths[three_roots_mask] + eps)).float()

            root3_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_3).float()
            root3_mask *= (root_3 < (input_right_cumwidths[three_roots_mask] + eps)).float()

            roots = torch.stack([root_1, root_2, root_3], dim=-1)
            masks = torch.stack([root1_mask, root2_mask, root3_mask], dim=-1)
            mask_index = torch.argsort(masks, dim=-1, descending=True)[..., 0][..., None]
            outputs[three_roots_mask] = torch.gather(roots, dim=-1, index=mask_index).view(-1)

            # Deal with a -> 0 (almost quadratic) cases.

            quadratic_mask = inputs_a.abs() < quadratic_threshold
            a = inputs_b[quadratic_mask]
            b = inputs_c[quadratic_mask]
            c = (inputs_d[quadratic_mask] - inputs[quadratic_mask])
            alpha = (-b + torch.sqrt(b.pow(2) - 4*a*c)) / (2*a)
            outputs[quadratic_mask] = alpha + input_left_cumwidths[quadratic_mask]

            shifted_outputs = (outputs - input_left_cumwidths)
            logabsdet = -torch.log((3 * inputs_a * shifted_outputs.pow(2) +
                                    2 * inputs_b * shifted_outputs +
                                    inputs_c))
        else:
            shifted_inputs = (inputs - input_left_cumwidths)
            outputs = (inputs_a * shifted_inputs.pow(3) +
                       inputs_b * shifted_inputs.pow(2) +
                       inputs_c * shifted_inputs +
                       inputs_d)

            logabsdet = torch.log((3 * inputs_a * shifted_inputs.pow(2) +
                                   2 * inputs_b * shifted_inputs +
                                   inputs_c))

        logabsdet = torch.sum(logabsdet, dim=1)

        if rev:
            outputs = outputs * (right - left) + left
            logabsdet = logabsdet - math.log(top - bottom) + math.log(right - left)
        else:
            outputs = outputs * (top - bottom) + bottom
            logabsdet = logabsdet + math.log(top - bottom) - math.log(right - left)
        masked_outputs[inside_interval_mask], masked_logabsdet[inside_interval_mask] = outputs, logabsdet

        return masked_outputs, masked_logabsdet

    def searchsorted(self, bin_locations, inputs, eps=1e-6):
        bin_locations[..., -1] += eps
        return torch.sum(inputs[..., None] >= bin_locations,dim=-1) - 1

    def cbrt(self, x):
        """Cube root. Equivalent to torch.pow(x, 1/3), but numerically stable."""
        return torch.sign(x) * torch.exp(torch.log(torch.abs(x)) / 3.0)


    def _permute(self, x, rev=False):
        '''Performs the permutation and scaling after the coupling operation.
        Returns transformed outputs and the LogJacDet of the scaling operation.'''

        scale = torch.ones(x.shape[-1]).to(x.device)
        perm_log_jac = torch.sum(-torch.log(scale))

        if rev:
            return (self.permute_function(x * scale, self.w_perm_inv),
                    perm_log_jac)
        else:
            return (self.permute_function(x, self.w_perm) / scale,
                    perm_log_jac)

    def forward(self, x, c=[], rev=False, jac=True):
        """
        :param x: Input
        :param c: Condition
        :param rev: If True, invert (sampling)
        :param jac: Don't know
        :return: log of determinate of Jacobian
        """
        self.bounds = self.bounds.to(x[0].device)
        if rev:
            x, global_scaling_jac = self._permute(x[0], rev=True)
            x = (x,)
        # get two coupling blocks
        x1, x2 = torch.split(x[0], self.splits, dim=1)

        if self.conditional:
            x1c = torch.cat([x1, *c], 1)
        else:
            x1c = x1

        if not rev:
            # Transform first coupling block with subnet
            theta = self.subnet(x1c).reshape(x1c.shape[0], self.splits[1], 2*self.num_bins + 2)
            # Use transformed coupling block and second coupling block to feed to your spline function
            x2, j2 = self._unconstrained_cubic_spline(x2, theta, rev=False)
        else:
            theta = self.subnet(x1c).reshape(x1c.shape[0], self.splits[1], 2*self.num_bins + 2)
            x2, j2 = self._unconstrained_cubic_spline(x2, theta, rev=True)

        log_jac_det = j2
        x_out = torch.cat((x1, x2), 1)
        if not rev:
            x_out, global_scaling_jac = self._permute(x_out, rev=False)

        # add the global scaling Jacobian to the total.
        # trick to get the total number of non-channel dimensions:
        # number of elements of the first channel of the first batch member
        n_pixels = x_out[0, :1].numel()
        log_jac_det += (-1)**rev * n_pixels * global_scaling_jac
        return (x_out,), log_jac_det

    def output_dims(self, input_dims):
        return input_dims


class QuadraticSplineBlock(InvertibleModule):
    BIN_WIDTH_MIN = 1e-3
    BIN_HEIGHT_MIN = 1e-3
    DELTA_MIN = 1e-3

    def _softplus_with_min(self, x, min=1e-3):
        # give one when x is zero, give min when softplus(x) is zero
        return min + (1 - min) * F.softplus(x) / math.log(2)

    def _softmax_with_min(self, x, min=1e-3):
        # give min when any softmax(x) is zero but still sum to one
        return min + (1 - min * x.shape[-1]) * F.softmax(x, dim=-1)

    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor=None, perm=None, num_bins=10,
                 bounds_init=1.):
        super().__init__(dims_in, dims_c)

        # only one dimensional data is supported (same for condition)
        assert len(dims_in) == 1 and len(dims_in[0]) == 1
        self.channel_count = dims_in[0][0]

        if len(dims_c) == 0:
            self.condition_channels = 0
            self.conditional = False
        else:
            assert len(dims_c[0]) == 1
            self.condition_channels = dims_c[0][0]
            self.conditional = True

        self.splits = (self.channel_count - self.channel_count // 2, self.channel_count // 2)

        # declaring as parameter makes sense for subtle reasons like automatically
        # moving to right device and updating when setting the state dict

        self.apply_spline_funcs = self._apply_splin
        self.unpack_spline_params_funcs = self._unpack_spline_params

        # contrary to https://arxiv.org/abs/1906.04032, we have 3 * bin_count + 1
        # instead of 3 * bin_count - 1, because we also want to learn derivatives at
        # the boundary; we don't care about discontinuities if our data is (mostly)
        # in bounds, and otherwise you have to fix something anyway
        assert not subnet_constructor is None

        self.subnet = subnet_constructor(self.splits[0] + self.condition_channels,
                                         (3 * num_bins + 1) * self.splits[1])

        if perm is None:
            perm = np.random.permutation(self.channel_count)

        channel_perm = np.zeros((self.channel_count, self.channel_count))
        for i, j in enumerate(perm):
            channel_perm[i, j] = 1.

        channel_perm = torch.FloatTensor(channel_perm)
        self.channel_perm = nn.Parameter(channel_perm, requires_grad=False)
        channel_inv_perm = torch.FloatTensor(channel_perm.T)
        self.channel_inv_perm = nn.Parameter(channel_inv_perm, requires_grad=False)

        self.bin_count = num_bins
        self.bound = bounds_init

    def spline(self, xy, x_knots, y_knots, bin_widths, bin_heights,
               knot_slopes, inverse=False):
        bin_count = bin_widths.shape[-1]

        xy_knots = (x_knots, y_knots)[inverse]
        # make x contiguous here (it becomes non-contiguous from the split in
        # forward), otherwise we get a performance warning
        bin_indices = torch.searchsorted(xy_knots,
                                         xy[..., None].contiguous()).squeeze(-1)
        # searchsorted returns the index of the knot after the value of xy, but
        # we want the knot before xy
        bin_indices = bin_indices - 1

        in_bounds = (bin_indices >= 0) & (bin_indices < bin_count)
        bin_indices_in_bounds = bin_indices[in_bounds]

        x1 = x_knots[in_bounds, bin_indices_in_bounds]
        y1 = y_knots[in_bounds, bin_indices_in_bounds]
        x_width = bin_widths[in_bounds, bin_indices_in_bounds]
        y_width = bin_heights[in_bounds, bin_indices_in_bounds]
        delta_1 = knot_slopes[in_bounds, bin_indices_in_bounds]
        delta_2 = knot_slopes[in_bounds, bin_indices_in_bounds + 1]

        x_left_bound = x_knots[~in_bounds, [0]]
        x_right_bound = x_knots[~in_bounds, [-1]]
        y_bottom_bound = y_knots[~in_bounds, [0]]
        y_top_bound = y_knots[~in_bounds, [-1]]

        yx = torch.full_like(xy, np.nan)
        log_jac_diag = torch.full_like(xy, 0)  # jac_diag is one when out of bounds
        yx[~in_bounds] = self.linear(xy[~in_bounds], x_left_bound, x_right_bound,
                                y_bottom_bound, y_top_bound, inverse=inverse)
        yx[in_bounds], log_jac_diag[in_bounds] = self.rational_quadratic(xy[in_bounds],
                                                                    x1, y1, x_width, y_width, delta_1, delta_2,
                                                                    inverse=inverse)
        log_jac_det = torch.sum(log_jac_diag, dim=-1)

        return yx, log_jac_det

    def rational_quadratic(self, xy, x1, y1, x_width, y_width,
                           delta_1, delta_2, inverse=False):
        # Compute rational quadratic spline between (x1, y1) and (x1 + x_width,
        # y1 + y_width) with derivatives delta_1 and delta_2 at these points, or compute
        # the inverse; compare https://arxiv.org/abs/1906.04032, eqs. (4)-(8)
        s = y_width / x_width

        if not inverse:
            x = xy

            eta = (x - x1) / x_width
            rev_eta = 1 - eta
            eta_rev_eta = eta * rev_eta

            y_denom = (s + (delta_1 + delta_2 - 2 * s) * eta_rev_eta)
            y = y1 + y_width * (s * eta ** 2 + delta_1 * eta_rev_eta) / y_denom
            yx = y

            # compute jacobian
            jac_numer = s ** 2 * (delta_2 * eta ** 2
                                  + 2 * s * eta_rev_eta + delta_1 * rev_eta ** 2)

            # jac_diag = jac_diag_numer / jac_diag_denom,
            # where jac_diag_denom = y_denom**2
            log_jac = torch.log(jac_numer) - 2 * torch.log(y_denom)
        else:
            y = xy

            y_shifted = y - y1
            shifted_delta_sum = (delta_2 + delta_1 - 2 * s)

            a = y_width * (s - delta_1) + y_shifted * shifted_delta_sum
            b = y_width * delta_1 - y_shifted * shifted_delta_sum
            c = -s * y_shifted

            eta = 2 * c / (-b - torch.sqrt(b ** 2 - 4 * a * c))
            rev_eta = 1 - eta
            eta_rev_eta = eta * rev_eta

            x = x1 + x_width * eta
            yx = x

            # compute jacobian identically to non-inverse case, just with different
            # optimisations
            inv_jac_numer = s ** 2 * (delta_2 * eta ** 2
                                      + 2 * s * eta_rev_eta + delta_1 * rev_eta ** 2)
            inv_jac_denom = (s + shifted_delta_sum * eta_rev_eta) ** 2

            log_inv_jac = torch.log(inv_jac_numer) - torch.log(inv_jac_denom)
            log_jac = -log_inv_jac

        return yx, log_jac

    def linear(self,xy, x1, x2, y1, y2, inverse=False):
        if not inverse:
            x = xy
            return (y2 - y1) / (x2 - x1) * (x - x1) + y1
        else:
            y = xy
            return (x2 - x1) / (y2 - y1) * (y - y1) + x1

    def _unpack_spline_params(self, theta):
        bin_widths_offset, bin_heights_offset, knot_slopes_offset \
            = 0, self.bin_count, 2 * self.bin_count

        bin_widths_unconstr = theta[..., bin_widths_offset: bin_heights_offset]
        bin_heights_unconstr = theta[..., bin_heights_offset: knot_slopes_offset]
        knot_slopes_unconstr = theta[..., knot_slopes_offset:]

        bin_widths = 2 * self.bound * self._softmax_with_min(
            bin_widths_unconstr, min=self.BIN_HEIGHT_MIN)
        bin_heights = 2 * self.bound * self._softmax_with_min(
            bin_heights_unconstr, min=self.BIN_WIDTH_MIN)
        knot_slopes = self._softplus_with_min(
            knot_slopes_unconstr, min=self.DELTA_MIN)

        # cumsum starts at bin_widths[..., 0], but we want to start at zero
        x_knots = F.pad(torch.cumsum(bin_widths, dim=-1),
                          (1, 0), value=0) - self.bound
        y_knots = F.pad(torch.cumsum(bin_heights, dim=-1),
                          (1, 0), value=0) - self.bound

        return dict(x_knots=x_knots, y_knots=y_knots, bin_widths=bin_widths,
                    bin_heights=bin_heights, knot_slopes=knot_slopes)

    def _apply_spline(self, xy, x_knots, y_knots,
                           bin_widths, bin_heights, knot_slopes, inverse=False):
        return self.spline(xy, x_knots, y_knots, bin_widths, bin_heights,
                      knot_slopes, inverse=inverse)


    def forward(self, x, c=[], rev=False, jac=True):
        x, = x

        if rev:
            x = F.linear(x, self.channel_inv_perm)

        x1, x2 = torch.split(x, self.splits, dim=-1)

        if self.conditional:
            x1_with_cond = torch.cat([x1, *c], dim=1)
        else:
            x1_with_cond = x1

        # we want to compute theta for each channel of the second split, but the
        # output of the subnet will be a flattened version of theta, cramming
        # together the channel and parameter dimension of theta; so reshape
        theta = self.subnet(x1_with_cond)
        theta = theta.reshape(-1, self.splits[1], 3 * self.bin_count + 1)

        x2, log_jac_det_2 = self._apply_spline(x2,theta, inverse=rev)

        log_jac_det = log_jac_det_2
        x_out = torch.cat((x1, x2), dim=1)

        if not rev:
            x_out = F.linear(x_out, self.channel_perm)

        return (x_out,), log_jac_det

    def output_dims(self, input_dims):
        return input_dims


def create_fully_connected_net(dims_in, dims_out, dims_intern, num_layers, dropout, bayesian, prior_prec):
    assert num_layers >= 2

    Linear = partial(VBLinear, prior_prec=prior_prec) \
    if bayesian else nn.Linear
    use_dropout = False if dropout == 0.0 else True

    layers = []

    layers.append(Linear(dims_in, dims_intern))
    if use_dropout:
        layers.append(nn.Dropout(p=dropout))

    layers.append(nn.ReLU())

    for n in range(num_layers-2):
        layers.append(Linear(dims_intern,dims_intern))
        if use_dropout:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.ReLU())

    layers.append(Linear(dims_intern, dims_out))

    return nn.Sequential(*layers)
