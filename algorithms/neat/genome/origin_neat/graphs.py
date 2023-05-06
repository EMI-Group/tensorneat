"""Directed graph algorithm implementations."""


def creates_cycle(connections, test):
    """
    Returns true if the addition of the 'test' connection would create a cycle,
    assuming that no cycle already exists in the graph represented by 'connections'.
    """
    i, o = test
    if i == o:
        return True

    visited = {o}
    while True:
        num_added = 0
        for a, b in connections:
            if a in visited and b not in visited:
                if b == i:
                    return True

                visited.add(b)
                num_added += 1

        if num_added == 0:
            return False


def required_for_output(inputs, outputs, connections):
    """
    Collect the nodes whose state is required to compute the final network output(s).
    :param inputs: list of the input identifiers
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    NOTE: It is assumed that the input identifier set and the node identifier set are disjoint.
    By convention, the output node ids are always the same as the output index.

    Returns a set of identifiers of required nodes.
    """
    assert not set(inputs).intersection(outputs)

    required = set(outputs)
    s = set(outputs)
    while 1:
        # Find nodes not in s whose output is consumed by a node in s.
        t = set(a for (a, b) in connections if b in s and a not in s)

        if not t:
            break

        layer_nodes = set(x for x in t if x not in inputs)
        if not layer_nodes:
            break

        required = required.union(layer_nodes)
        s = s.union(t)

    return required


def feed_forward_layers(inputs, outputs, connections):
    """
    Collect the layers whose members can be evaluated in parallel in a feed-forward network.
    :param inputs: list of the network input nodes
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.

    Returns a list of layers, with each layer consisting of a set of node identifiers.
    Note that the returned layers do not contain nodes whose output is ultimately
    never used to compute the final network output.
    """

    required = required_for_output(inputs, outputs, connections)

    layers = []
    s = set(inputs)
    while 1:
        # Find candidate nodes c for the next layer.  These nodes should connect
        # a node in s to a node not in s.
        c = set(b for (a, b) in connections if a in s and b not in s)
        # Keep only the used nodes whose entire input set is contained in s.
        t = set()
        for n in c:
            if n in required and all(a in s for (a, b) in connections if b == n):
                t.add(n)

        if not t:
            break

        layers.append(t)
        s = s.union(t)

    return layers


def node_calculate_sequence(inputs, outputs, connections):
    """
    Collect the sequence of nodes to calculate in order to compute the final network output(s).
    :param required_nodes:
    :param connections:
    :return:
    """
    required_nodes = required_for_output(inputs, outputs, connections)
    useful_nodes = required_nodes.copy()
    useful_nodes.update(inputs)
    useful_connections = [c for c in connections if c[0] in useful_nodes and c[1] in useful_nodes]

    # do topological sort on useful_connections
    in_degrees = {n: 0 for n in useful_nodes}
    for a, b in useful_connections:
        in_degrees[b] += 1
    topological_order = []
    while len(topological_order) < len(useful_nodes):
        for n in in_degrees:
            if in_degrees[n] == 0:
                topological_order.append(n)
                in_degrees[n] -= 1
                for a, b in useful_connections:
                    if a == n:
                        in_degrees[b] -= 1

    [topological_order.remove(n) for n in inputs]  # remove inputs from topological order
    return topological_order, useful_connections


if __name__ == '__main__':
    inputs = [-1, -2]
    outputs = [0]
    connections = [(-2, 2), (-2, 3), (4, 0), (3, 0), (2, 0), (2, 3), (2, 4)]
    seqs = node_calculate_sequence(inputs, outputs, connections)
    print(seqs)
