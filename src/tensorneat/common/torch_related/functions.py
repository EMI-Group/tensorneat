import torch

SCALE = 3

def torch_scaled_sigmoid(z):
    return (1 / (1 + torch.exp(-z))) * SCALE

def torch_sigmoid(z):
    return 1 / (1 + torch.exp(-z))

def torch_scaled_tanh(z):
    return torch.tanh(z) * SCALE

def torch_tanh(z):
    return torch.tanh(z)

def torch_sin(z):
    return torch.sin(z)

def torch_relu(z):
    return torch.relu(z)

def torch_lelu(z):
    leaky = 0.005
    return torch.where(z > 0, z, leaky * z)

def torch_identity(z):
    return z

def torch_inv(z):
    z = torch.where(z == 0, torch.tensor(1e-7, dtype=z.dtype, device=z.device), z)
    return 1 / z

def torch_log(z):
    z = torch.clamp(z, min=1e-7)
    return torch.log(z)

def torch_exp(z):
    return torch.exp(z)

def torch_abs(z):
    return torch.abs(z)

def torch_sum(z):
    return torch.nansum(z, dim=1)

def torch_product(z):
    return torch.nanprod(z, dim=1)

def torch_max(z):
    return torch.nanmax(z, dim=1).values

def torch_min(z):
    return torch.nanmin(z, dim=1).values

def torch_maxabs(z):
    z = torch.where(torch.isnan(z), torch.tensor(0.0, dtype=z.dtype, device=z.device), z)
    abs_z = torch.abs(z)
    max_abs_indices = torch.argmax(abs_z, dim=1)
    return torch.gather(z, dim=1, index=max_abs_indices.unsqueeze(1)).squeeze(1)

def torch_mean(z):
    valid_count = torch.sum(~torch.isnan(z), dim=1)
    return torch.nansum(z, dim=1) / valid_count

torch_functions = {
    "scaled_sigmoid": torch_scaled_sigmoid,
    "sigmoid": torch_sigmoid,
    "scaled_tanh": torch_scaled_tanh,
    "tanh": torch_tanh,
    "sin": torch_sin,
    "relu": torch_relu,
    "lelu": torch_lelu,
    "identity": torch_identity,
    "inv": torch_inv,
    "log": torch_log,
    "exp": torch_exp,
    "abs": torch_abs,
    "sum": torch_sum,
    "product": torch_product,
    "max": torch_max,
    "min": torch_min,
    "maxabs": torch_maxabs,
    "mean": torch_mean,
}