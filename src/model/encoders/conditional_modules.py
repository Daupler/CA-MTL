import torch
import numbers
from typing import List, Optional, Union
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """ Feature-wise Linear Modulation (FiLM) layer"""
    def __init__(self, input_size, output_size, num_film_layers=1, layer_norm=False):
        """
        :param input_size: feature size of x_cond
        :param output_size: feature size of x_to_film
        :param layer_norm: true or false
        """
        super(FiLM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_film_layers = num_film_layers
        self.layer_norm = nn.LayerNorm(output_size) if layer_norm else None
        film_output_size = self.output_size * num_film_layers * 2
        self.gb_weights = nn.Linear(self.input_size, film_output_size)
        self.gb_weights.bias.data.fill_(0)

    def forward(self, x_cond, x_to_film):
        gb = self.gb_weights(x_cond).unsqueeze(1)
        gamma, beta = torch.chunk(gb, 2, dim=-1)
        out = (1 + gamma) * x_to_film + beta
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        return out


class CBDA(nn.Module):
    """ Conditional Block Diagonal Attention (CBDA) layer"""
    def __init__(self, input_size, output_size, blocks=1, num_film_layers=1, layer_norm=False):
        """
        :param input_size: feature size of x_cond
        :param output_size: feature size of x_to_film
        :param layer_norm: true or false
        """
        super(CBDA, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_film_layers = num_film_layers
        self.layer_norm = nn.LayerNorm(output_size) if layer_norm else None
        self.blocks = blocks
        film_output_size = self.output_size * num_film_layers * 2
        self.gb_weights = nn.Linear(self.input_size, film_output_size)
        self.gb_weights.bias.data.fill_(0)
        
        # TorchScript does not support variable length arguments which result in this 
        # calculation. To address this in the short term, I am fixing the allowable inputs
        # to the hidden_size and max_seq_length since they determine exclusively the size of the
        # tensors here
        assert (self.input_size==768) & (self.output_size==86) & (self.blocks==3),"CBDA is only applicable right now with max_seq_length=256 and hidden_size=768"

    def forward(self, x_cond: torch.Tensor, x_to_film: torch.Tensor):
        gb = self.gb_weights(x_cond).unsqueeze(1)
        gamma, beta = torch.chunk(gb, 2, dim=-1)
        out = (1 + gamma) * x_to_film + beta
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        
        # breaking up the block diagonal process here from the nice little list comprehension
        # that TorchScript does not like, into a control structure and explicitly unpacking
        # the individual chunks to pass them to the block_diag method since the expand operator
        # is not supported with TorchScript
        block_diag_out = []
        for out_b in out:
            chunks = list(out_b.chunk(self.blocks, 0))
            block_diag_out.append(torch.block_diag(chunks[0], chunks[1], chunks[2]))
        #out = [torch.block_diag(*list(out_b.chunk(self.blocks, 0))) for out_b in out]
        out = torch.stack(block_diag_out)
        return out[:, :, :out.size(1)]


class ConditionalLayerNorm(nn.Module):
    r"""Applies Conditional Layer Normalization over a mini-batch of inputs.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma(z) + \beta(z)

    The mean and standard-deviation are calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine`, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes. The affine transformation is molulated by a conditional tensor.
    In our case, we use task embeddings z.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> input_ = torch.randn(20, 5, 10, 10)
        >>> condition = torch.randn(20, 10)
        >>> # With Learnable Parameters
        >>> m = ConditionalLayerNorm([10, 10])
        >>> # Normalize over last dimension of size 10
        >>> m = nn.LayerNorm(10)
        >>> # Activating the module
        >>> output = m(input_, condition)

    .. _`Layer Normalization`: https://arxiv.org/abs/1607.06450
    .. _`Conditional Layer Normalization`: https://arxiv.org/
    """
    __constants__ = ['normalized_shape', 'condition_size', 'weight', 'bias', 'eps']

    def __init__(self, normalized_shape, condition_size, eps=1e-5):
        super(ConditionalLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)

        self.condition_size = condition_size
        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
        self.ln_weight_modulation = FiLM(condition_size, sum(normalized_shape))
        self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input_, condition, task_id):
        unique_task_ids = torch.unique(task_id)
        cln_output = torch.zeros_like(input_)
        for unique_task_id in unique_task_ids:
            task_id_filter = task_id == unique_task_id
            task_emb = condition[task_id_filter][0].unsqueeze(0)
            weight = self.ln_weight_modulation(task_emb, self.weight).view(-1)
            cln_output[task_id_filter] = F.layer_norm(input_[task_id_filter], self.normalized_shape, weight, self.bias, self.eps)
        return cln_output

    def extra_repr(self):
        return '{normalized_shape}, {condition_size}, eps={eps}'.format(**self.__dict__)


class ConditionalBottleNeck(nn.Module):
    """Down projection and up projection with FiLM layers within Transformer layer."""
    def __init__(self, config):
        super(ConditionalBottleNeck, self).__init__()
        self.emb_transf = nn.Linear(config.hidden_size, config.hidden_size)
        self.hidden_modulation = FiLM(config.hidden_size, config.hidden_size)
        self.down_proj_layer = nn.Linear(config.hidden_size, config.hidden_size//3)
        self.up_proj_layer = nn.Linear(config.hidden_size//3, config.hidden_size)

    def forward(self, x_cond, hidden_states):
        x_cond = self.emb_transf(x_cond)
        hidden_states = self.hidden_modulation(x_cond=x_cond, x_to_film=hidden_states)
        hidden_states = self.down_proj_layer(hidden_states)
        hidden_states = self.up_proj_layer(hidden_states)
        return hidden_states
