from typing import Tuple
from random import gauss, choice, random


def clip(x, min_val, max_val):
    return min(max(x, min_val), max_val)


class NodeGene:

    def __init__(self, key: int, config, init_val=True):
        self.key = key
        self.config = config

        if init_val:
            self.init_value()
        else:
            self.bias = 0
            self.response = 1
            self.act = 0
            self.agg = 0

    def __repr__(self):
        return f'node({self.key}, bias: {self.bias:.3f}, ' \
               f'response: {self.response:.3f}, act: {self.act}, agg: {self.agg})'

    def __eq__(self, other):
        if not isinstance(other, NodeGene):
            return False
        return self.key == other.key and \
            self.bias == other.bias and \
            self.response == other.response and \
            self.act == other.act and \
            self.agg == other.agg

    def copy(self):
        new_gene = self.__class__(self.key, config=self.config, init_val=False)
        new_gene.bias = self.bias  # numpy array is mutable, so we need to copy it
        new_gene.response = self.response
        new_gene.act = self.act
        new_gene.agg = self.agg
        return new_gene

    def init_value(self):
        c = self.config.gene
        self.bias = gauss(c.bias.init_mean, c.bias.init_stdev)
        self.response = gauss(c.response.init_mean, c.response.init_stdev)
        self.act = choice(c.activation.options)
        self.agg = choice(c.aggregation.options)

        self.bias = clip(self.bias, c.bias.min_value, c.bias.max_value)
        self.response = clip(self.response, c.response.min_value, c.response.max_value)

    def distance(self, other):
        s = abs(self.bias - other.bias) + abs(self.response - other.response)
        if self.act != other.act:
            s += 1
        if self.agg != other.agg:
            s += 1
        return s

    def mutate(self):
        self.bias = mutate_float(self.bias, self.config.gene.bias)
        self.response = mutate_float(self.response, self.config.gene.response)
        self.act = mutate_string(self.act, self.config.gene.activation)
        self.agg = mutate_string(self.agg, self.config.gene.aggregation)

    @classmethod
    def crossover(cls, g1, g2):
        assert g1.key == g2.key
        c = cls(g1.key, g1.config, init_val=False)
        c.bias = g1.bias if random() > 0.5 else g2.bias
        c.response = g1.response if random() > 0.5 else g2.response
        c.act = g1.act if random() > 0.5 else g2.act
        c.agg = g1.agg if random() > 0.5 else g2.agg
        return c


class ConnectionGene:
    def __init__(self, key: Tuple[int, int], config, init_val=True):
        self.key = key
        self.config = config
        self.enabled = True
        if init_val:
            self.init_value()
        else:
            self.weight = 1

    def __repr__(self):
        return f'connection({self.key}, {self.weight:.3f}, {self.enabled})'

    def __eq__(self, other):
        if not isinstance(other, ConnectionGene):
            return False
        return self.key == other.key and \
            self.weight == other.weight and \
            self.enabled == other.enabled

    def copy(self):
        new_gene = self.__class__(self.key, self.config, init_val=False)
        new_gene.weight = self.weight
        new_gene.enabled = self.enabled
        return new_gene

    def init_value(self):
        c = self.config.gene
        self.weight = gauss(c.weight.init_mean, c.weight.init_stdev)
        self.weight = clip(self.weight, c.weight.min_value, c.weight.max_value)

    def distance(self, other):
        s = abs(self.weight - other.weight)
        if self.enabled != other.enabled:
            s += 1
        return s

    def mutate(self):
        self.weight = mutate_float(self.weight, self.config.gene.weight)
        if random() < self.config.gene.enabled.mutate_rate:
            self.enabled = not self.enabled

    @classmethod
    def crossover(cls, g1, g2):
        assert g1.key == g2.key
        c = cls(g1.key, g1.config, init_val=False)
        c.weight = g1.weight if random() > 0.5 else g2.weight
        c.enabled = g1.enabled if random() > 0.5 else g2.enabled
        return c


# HAHA, exactly the bug is here!!
# After I fixed it, the result is much better!!
def mutate_float(v, vc):
    """vc -> value config"""
    r = random()
    if r < vc.mutate_rate:
        v += gauss(0, vc.mutate_power)
        v = clip(v, vc.min_value, vc.max_value)
    # Previous, seems like a huge bug
    # if r < vc.mutate_rate + vc.replace_rate:
    # Now:
    elif r < vc.mutate_rate + vc.replace_rate:
        v = gauss(vc.init_mean, vc.init_stdev)
        v = clip(v, vc.min_value, vc.max_value)
    return v


def mutate_string(v, vc):
    """vc -> value config"""
    r = random()
    if r < vc.mutate_rate:
        v = choice(vc.options)
    return v
