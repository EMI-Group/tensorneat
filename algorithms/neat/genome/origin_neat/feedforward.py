from .graphs import node_calculate_sequence
from .activations import activation_dict
from .aggregations import aggregation_dict


class FeedForwardNetwork(object):
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key, 0.0) for key in inputs + outputs)

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            node_inputs = []
            for i, w in links:
                node_inputs.append(self.values[i] * w)
            if len(node_inputs) == 0:
                s = 0.0
            else:
                s = agg_func(node_inputs)
            self.values[node] = act_func(bias + response * s)

        return [self.values[i] for i in self.output_nodes]

    @staticmethod
    def create(genome):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Gather expressed connections.
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]

        seqs, useful_connections = node_calculate_sequence(genome.input_keys, genome.output_keys, connections)
        node_evals = []
        for node in seqs:
            inputs = []
            for conn_key in useful_connections:
                inode, onode = conn_key
                if onode == node:
                    cg = genome.connections[conn_key]
                    inputs.append((inode, cg.weight))

            ng = genome.nodes[node]
            act_func = activation_dict[ng.act]
            agg_func = aggregation_dict[ng.agg]
            node_evals.append((node, act_func, agg_func, ng.bias, ng.response, inputs))

        return FeedForwardNetwork(genome.input_keys, genome.output_keys, node_evals)
