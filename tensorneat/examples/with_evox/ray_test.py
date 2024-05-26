import ray
ray.init(num_gpus=2)

available_resources = ray.available_resources()
print("Available resources:", available_resources)
