from random import random, choice

from .gene import NodeGene, ConnectionGene
from .graphs import creates_cycle


class Genome:
    def __init__(self, key, config, global_idx, init_val=True):
        # Unique identifier for a genome instance.
        self.key = key
        self.config = config
        self.global_idx = global_idx

        # (gene_key, gene) pairs for gene sets.
        self.connections = {}
        self.nodes = {}

        # Fitness results.
        self.fitness = None

        # self.input_keys = [-i - 1 for i in range(config.basic.num_inputs)]
        # self.output_keys = [i for i in range(config.basic.num_outputs)]

        if init_val:
            self.initialize()

    def __repr__(self):
        nodes_info = ',\n\t\t'.join(map(str, self.nodes.values()))
        connections_info = ',\n\t\t'.join(map(str, self.connections.values()))

        return f'Genome(\n\t' \
               f'key: {self.key}, \n' \
               f'\tinput_keys: {self.input_keys}, \n' \
               f'\toutput_keys: {self.output_keys}, \n' \
               f'\tnodes: \n\t\t' \
               f'{nodes_info} \n' \
               f'\tconnections: \n\t\t' \
               f'{connections_info} \n)'

    def __eq__(self, other):
        if not isinstance(other, Genome):
            return False
        if self.key != other.key:
            return False
        if len(self.nodes) != len(other.nodes) or len(self.connections) != len(other.connections):
            return False
        for k, v in self.nodes.items():
            o_v = other.nodes.get(k)
            if o_v is None or v != o_v:
                return False
        for k, v in self.connections.items():
            o_v = other.connections.get(k)
            if o_v is None or v != o_v:
                return False
        return True

    def initialize(self):
        """Configure a new genome based on the given configuration."""

        # Create node genes for the output pins.
        for node_key in self.output_keys:
            self.nodes[node_key] = NodeGene(node_key, self.config, init_val=True)

        # Add connections based on initial connectivity type.
        # ONLY ALLOW FULL HERE AND NO HIDDEN!!!
        for i in self.input_keys:
            for j in self.output_keys:
                key = (i, j)
                self.connections[key] = ConnectionGene(key, self.config, init_val=True)

    def distance(self, other):
        """Calculate the distance between two genomes."""

        wc = self.config.genome.compatibility_weight_coefficient
        dc = self.config.genome.compatibility_disjoint_coefficient

        node_distance = 0.0
        if self.nodes or other.nodes:  # otherwise, both are empty
            disjoint_nodes = 0
            for k2 in other.nodes:
                if k2 not in self.nodes:
                    disjoint_nodes += 1

            for k1, n1 in self.nodes.items():
                n2 = other.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    node_distance += n1.distance(n2)

            max_nodes = max(len(self.nodes), len(other.nodes))
            node_distance = (wc * node_distance + dc * disjoint_nodes) / max_nodes

        connection_distance = 0.0
        if self.connections or other.connections:
            disjoint_connections = 0
            for k2 in other.connections:
                if k2 not in self.connections:
                    disjoint_connections += 1

            for k1, c1 in self.connections.items():
                c2 = other.connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    # Homologous genes compute their own distance value.
                    connection_distance += c1.distance(c2)

            max_conn = max(len(self.connections), len(other.connections))
            connection_distance = (wc * connection_distance + dc * disjoint_connections) / max_conn

        return node_distance + connection_distance

    @classmethod
    def crossover(cls, new_key, g1, g2):
        if g1.fitness > g2.fitness:
            p1, p2 = g1, g2
        else:
            p1, p2 = g2, g1

        child = cls(new_key, p1.config, p1.global_idx, init_val=False)

        for k, cg1 in p1.connections.items():
            cg2 = p2.connections.get(k)
            if cg2 is None:
                child.connections[k] = cg1.copy()
            else:
                child.connections[k] = ConnectionGene.crossover(cg1, cg2)

        for k, ng1 in p1.nodes.items():
            ng2 = p2.nodes.get(k)
            if ng2 is None:
                child.nodes[k] = ng1.copy()
            else:
                child.nodes[k] = NodeGene.crossover(ng1, ng2)

        return child

    def mutate(self):
        c = self.config.genome

        if c.single_structural_mutation:
            div = max(1, c.conn_add_prob + c.conn_delete_prob + c.node_add_prob + c.node_delete_prob)
            r = random()

            if r < c.node_add_prob / div:
                self.mutate_add_node()
            elif r < (c.node_add_prob + c.node_delete_prob) / div:
                self.mutate_delete_node()
            elif r < (c.node_add_prob + c.node_delete_prob + c.conn_add_prob) / div:
                self.mutate_add_connection()
            elif r < (c.node_add_prob + c.node_delete_prob + c.conn_add_prob + c.conn_delete_prob) / div:
                self.mutate_delete_connection()
        else:
            if random() < c.node_add_prob:
                self.mutate_add_node()
            if random() < c.node_delete_prob:
                self.mutate_delete_node()
            if random() < c.conn_add_prob:
                self.mutate_add_connection()
            if random() < c.conn_delete_prob:
                self.mutate_delete_connection()

        for cg in self.connections.values():
            cg.mutate()

        for ng in self.nodes.values():
            ng.mutate()

    def mutate_add_node(self):
        # create a node from splitting a connection
        if not self.connections:
            return -1

        # Choose a random connection to split
        conn_to_split = choice(list(self.connections.values()))
        new_node_id = self.global_idx.next_node()
        ng = NodeGene(new_node_id, self.config, init_val=False)
        self.nodes[new_node_id] = ng

        # Create two new connections
        conn_to_split.enabled = False
        i, o = conn_to_split.key
        con1 = ConnectionGene((i, new_node_id), self.config, init_val=False)
        con2 = ConnectionGene((new_node_id, o), self.config, init_val=False)

        # The new node+connections have roughly the same behavior as
        # the original connection (depending on the activation function of the new node).
        con2.weight = conn_to_split.weight
        self.connections[con1.key] = con1
        self.connections[con2.key] = con2

        return 1

    def mutate_delete_node(self):
        # Do nothing if there are no non-output nodes.
        available_nodes = [k for k in self.nodes if k not in self.output_keys]
        if not available_nodes:
            return -1

        del_key = choice(available_nodes)
        connections_to_delete = set()
        for k, v in self.connections.items():
            if del_key in v.key:
                connections_to_delete.add(v.key)

        for key in connections_to_delete:
            del self.connections[key]

        del self.nodes[del_key]

        return del_key

    def mutate_add_connection(self):
        """
        Attempt to add a new connection, the only restriction being that the output
        node cannot be one of the network input pins.
        """
        possible_outputs = list(self.nodes)
        out_node = choice(possible_outputs)

        possible_inputs = possible_outputs + self.input_keys
        in_node = choice(possible_inputs)

        # in recurrent networks, the input node can be the same as the output node
        key = (in_node, out_node)
        if key in self.connections:
            self.connections[key].enabled = True
            return -1

        # if feedforward network, check if the connection creates a cycle
        if self.config.genome.feedforward and creates_cycle(self.connections.keys(), key):
            return -1

        cg = ConnectionGene(key, self.config, init_val=True)
        self.connections[key] = cg
        return key

    def mutate_delete_connection(self):
        if self.connections:
            key = choice(list(self.connections.keys()))
            del self.connections[key]

    def complexity(self):
        return len(self.connections) * 2 + len(self.nodes) * 4