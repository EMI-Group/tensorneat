class BaseMutation:
    def __call__(self, key, genome, nodes, conns, new_node_key):
        raise NotImplementedError