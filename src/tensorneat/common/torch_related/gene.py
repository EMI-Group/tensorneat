from typing import Union, Callable
import torch
import torch.nn as nn

from .functions import torch_functions

class BaseGeneTorch(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(inputs: torch.Tensor):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, gene_dict):
        g = cls(**gene_dict)
        setattr(g, "gene_dict", gene_dict)
        return g

    def get_dict(self):
        return self.gene_dict


class DefaultNodeGeneTorch(BaseGeneTorch):
    def __init__(
        self,
        res: float,  # response
        bias: float,
        agg: Union[str, Callable],  # aggregation
        act: Union[str, Callable],  # activation
        is_output_node=False,
        *args,
        **kwargs
    ):
        super().__init__()
        self.res = nn.Parameter(torch.tensor(float(res), dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(float(bias), dtype=torch.float32))
        self.agg = agg
        self.act = act
        self.is_output_node = is_output_node

        if isinstance(self.act, str):
            assert self.act in torch_functions, f"function name {self.act} is not in torch_functions"
            self.act = torch_functions[self.act]
        
        if isinstance(self.agg, str):
            assert self.agg in torch_functions, f"function name {self.agg} is not in torch_functions"
            self.agg = torch_functions[self.agg]


    def forward(self, inputs):
        assert inputs.dim() == 2, f"torch genome needs batch inputs, however, got shape {inputs.shape}"
        if self.is_output_node:  # no activation
            val = self.res * self.agg(inputs) + self.bias
        else:
            val = self.act(self.res * self.agg(inputs) + self.bias)
        return val


class BiasNodeGeneTorch(DefaultNodeGeneTorch):
    def __init__(
        self,
        bias: float,
        aggregation: Callable,
        activation: Callable,
        is_output_node=False,
        *args,
        **kwargs
    ):
        super().__init__(1, bias, aggregation, activation, is_output_node)


class DefaultConnTorch(BaseGeneTorch):
    def __init__(self, weight: float, *args, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(weight), dtype=torch.float32))


    def forward(self, inputs):
        return inputs * self.weight

torch_genes = {
    "DefaultNode": DefaultNodeGeneTorch,
    "BiasNode": BiasNodeGeneTorch,
    "DefaultConn": DefaultConnTorch
}