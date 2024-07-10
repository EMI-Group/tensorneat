import networkx as nx
import matplotlib.pyplot as plt

# 创建一个空白的有向图
G = nx.DiGraph()

# 添加边
G.add_edge('A', 'B')
G.add_edge('A', 'C')
G.add_edge('B', 'C')
G.add_edge('C', 'D')

# 绘制有向图
