from tensorneat.common import StatefulBaseClass


class BaseSubstrate(StatefulBaseClass):

    connection_type = None

    def make_nodes(self, query_res):
        raise NotImplementedError

    def make_conns(self, query_res):
        raise NotImplementedError

    @property
    def query_coors(self):
        raise NotImplementedError

    @property
    def num_inputs(self):
        raise NotImplementedError

    @property
    def num_outputs(self):
        raise NotImplementedError

    @property
    def nodes_cnt(self):
        raise NotImplementedError

    @property
    def conns_cnt(self):
        raise NotImplementedError
