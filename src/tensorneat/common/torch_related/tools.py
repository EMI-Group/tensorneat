import torch.nn as nn


class StringNameModuleDict(nn.ModuleDict):
    def __init__(self, d=None):
        super().__init__()
        self.origin2str = {}
        self.str2origin = {}

        if d:
            for k, v in d.items():
                self[k] = v

    def __getitem__(self, key):
        return super().__getitem__(str(key))

    def __setitem__(self, key, module):
        str_key = str(key)
        super().__setitem__(str_key, module)
        self.origin2str[key] = str_key
        self.str2origin[str_key] = key

    def __delitem__(self, key):
        str_key = str(key)
        super().__delitem__(str_key)
        del self.origin2str[key]
        del self.str2origin[str_key]

    def __iter__(self):
        return iter(self.origin2str.keys())

    def __contains__(self, key):
        return str(key) in self.str2origin
