import torch
import torch.nn as nn

from typing import Dict, Tuple, List, Callable
from .gene import BaseGeneTorch, torch_genes
from .tools import StringNameModuleDict
from tensorneat.common.graph import topological_sort_python


class DefaultGenomeTorch(nn.Module):
    def __init__(
        self,
        nodes: Dict[int, BaseGeneTorch],
        conns: Dict[Tuple[int, int], BaseGeneTorch],
        input_idx: List[int],
        output_idx: List[int],
        topo_order: List[int] = None,
        output_transform: Callable = None,
    ):
        super().__init__()
        self.nodes = StringNameModuleDict(nodes)
        self.conns = StringNameModuleDict(conns)
        self.input_idx = input_idx
        self.output_idx = output_idx
        self.output_transform = output_transform

        self.topo_order = (
            topo_order
            if topo_order is not None
            else topological_sort_python(set(nodes.keys()), set(conns.keys()))
        )
        self.conns_activates_layers = []
        print([u for u in self.conns])
        for n in self.topo_order:
            if n in input_idx:
                self.conns_activates_layers.append([])
            else:
                activate_conns = [(in_, out) for in_, out in self.conns if out == n]
                self.conns_activates_layers.append(activate_conns)

    def forward(self, inputs):
        assert inputs.dim() == 2, "Batch inputs needed"
        assert inputs.shape[-1] == len(
            self.input_idx
        ), f"input shape should be (batch_size, {len(self.input_idx)}), but got {inputs.shape}"
        batch_size = inputs.shape[0]

        node_vals = torch.zeros((batch_size, len(self.nodes)), device=inputs.device, dtype=torch.float32)

        node_vals = node_vals.clone()
        node_vals[:, self.input_idx] = torch.as_tensor(inputs, dtype=torch.float32, device=node_vals.device)

        # fill input vals
        for input_idx, node_idx in enumerate(self.input_idx):
            node_vals[:, node_idx] = inputs[:, input_idx]

        for i, node_idx in enumerate(self.topo_order):
            if node_idx in self.input_idx:
                continue
            else:
                # vals_after_conns = []
                # for in_, out in self.conns_activates_layers[i]:
                #     vals_after_conns.append(
                #         self.conns[in_, out].forward(node_vals[in_])
                #     )

                # gather vals for node inputs
                vals_after_conns = torch.stack(
                    [
                        self.conns[in_, out].forward(node_vals[:, in_])
                        for in_, out in self.conns_activates_layers[i]
                    ],
                    axis=1,
                )
                
                # calculate node output and update node_vals
                node_vals = node_vals.clone()
                node_vals[:, node_idx] = self.nodes[node_idx].forward(vals_after_conns)

        output_vals = node_vals[:, self.output_idx]

        if self.output_transform is not None:
            output_vals = self.output_transform(output_vals)

        return output_vals

    @classmethod
    def from_dict(cls, genome_dict, **kwargs):
        genome_dict.update(kwargs)
        assert (
            genome_dict["genome_class"] == "DefaultGenome"
        ), "Only support DefaultGenome"
        assert (
            genome_dict["node_class"] in torch_genes
        ), f"{genome_dict['node_class']} not in torch_genes"
        assert (
            genome_dict["conn_class"] in torch_genes
        ), f"{genome_dict['conn_class']} not in torch_genes"

        node_gene_cls = torch_genes[genome_dict["node_class"]]
        conn_gene_cls = torch_genes[genome_dict["conn_class"]]

        nodes, conns = {}, {}
        for k, v in genome_dict["nodes"].items():
            nodes[k] = node_gene_cls.from_dict(v)
            if k in genome_dict["output_idx"]:
                nodes[k].is_output_node = True

        for k, v in genome_dict["conns"].items():
            conns[k] = conn_gene_cls.from_dict(v)

        return cls(
            nodes,
            conns,
            genome_dict["input_idx"],
            genome_dict["output_idx"],
            topo_order=genome_dict["topo_order"],
            output_transform=genome_dict["output_transform"],
        )
