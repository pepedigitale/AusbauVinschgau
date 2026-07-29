import networkx as nx
import pandas as pd
import numpy as np
from visualize_ean_plotly import plot_ean_plotly, show_ean

# Minimal nodesDf and edgesDf
nodes = pd.DataFrame({
    'pk_rel': [0.0, 10.0],
    'node_type': ['LdS', 'LdS']
}, index=['ME', 'MAL'])

edges = pd.DataFrame([
    {'node_from': 'ME', 'node_to': 'MAL'}
])

# Build simple graph
G = nx.DiGraph()

# Train 1: ME -> MAL
G.add_node('t1_dep', station='ME', time=8*3600, train='T1', event='dep')
G.add_node('t1_arr', station='MAL', time=9*3600, train='T1', event='arr')
G.add_edge('t1_dep', 't1_arr', kind='running')

# Train 2: MAL -> ME
G.add_node('t2_dep', station='MAL', time=8*3600+15*60, train='T2', event='dep')
G.add_node('t2_arr', station='ME', time=9*3600+15*60, train='T2', event='arr')
G.add_edge('t2_dep', 't2_arr', kind='running')

fig, ax = plot_ean_plotly(G, nodes, edges, title='Test EAN')

html_path = show_ean(fig, filename='test_ean_plot.html', auto_open=False)

print('WROTE', html_path)
with open(html_path, 'r', encoding='utf-8') as fh:
    txt = fh.read()

print('HAS_LAYER_CONTROLS:', 'layer-controls' in txt)
print('HAS_LAYER_JS:', 'function tracesFor' in txt)
