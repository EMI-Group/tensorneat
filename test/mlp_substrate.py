from tensorneat.algorithm.hyperneat.substrate.mlp import MLPSubstrate, analysis_substrate

layers = [3, 4, 2]
coor_range = (-1, 1, -1, 1)
nodes = analysis_substrate(layers, coor_range)
print(nodes)
