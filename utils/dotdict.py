# DotDict For Config. Case Insensitive.

class DotDict(dict):
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self["name"] = name

    def __getattr__(self, attr):
        attr = attr.lower()  # case insensitive
        if attr in self:
            return self[attr]
        else:
            raise AttributeError(f"'{self.__class__.__name__}-{self.name}' has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        attr = attr.lower()  # case insensitive
        if attr not in self:
            raise AttributeError(f"'{self.__class__.__name__}-{self.name}' has no attribute '{attr}'")
        self[attr] = value

    def __delattr__(self, attr):
        attr = attr.lower()  # case insensitive
        if attr in self:
            del self[attr]
        else:
            raise AttributeError(f"{self.__class__.__name__}-{self.name} object has no attribute '{attr}'")

    @classmethod
    def from_dict(cls, d, name):
        if not isinstance(d, dict):
            return d

        dot_dict = cls(name)
        for key, value in d.items():
            key = key.lower()  # case insensitive
            if isinstance(value, dict):
                dot_dict[key] = cls.from_dict(value, key)
            else:
                dot_dict[key] = value
                if dot_dict[key] == "True":  # Fuck! Json has no bool type!
                    dot_dict[key] = True
                if dot_dict[key] == "False":
                    dot_dict[key] = False
                if dot_dict[key] == "None":
                    dot_dict[key] = None
        return dot_dict


if __name__ == '__main__':
    nested_dict = {
        "a": 1,
        "b": {
            "c": 2,
            "ACDeef": {
                "e": 3
            }
        }
    }

    dd = DotDict.from_dict(nested_dict, "root")
    print(dd.b.acdeef.e)  # 输出：3
