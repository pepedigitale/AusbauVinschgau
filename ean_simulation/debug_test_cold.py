import pandas as pd
import networkx as nx
from visualize_ean_plotly import draw_ean_plotly, _station_x_map, _get_boundary_stations

# Load nodes.csv
nodesDf = pd.read_csv(r"../infra_data/nodes.csv", sep=';', decimal=',', index_col=0)

# Minimal graph with one train stopping at COLD
G = nx.DiGraph()
G.add_node(('T1', 'COLD', 'arr', 1), train='T1', station='COLD', event='arr', time=8*3600+600, seq=1)
G.add_node(('T1', 'COLD', 'dep', 1), train='T1', station='COLD', event='dep', time=8*3600+900, seq=1)
G.add_edge(('T1', 'COLD', 'arr', 1), ('T1', 'COLD', 'dep', 1), kind='dwell')

fig = __import__('plotly').graph_objs.Figure()

# Provide the minimal layout.meta expected by draw_ean_plotly when called
# directly (plot_ean_plotly normally sets this).
fig.update_layout(meta={
    'realization_count': 0,
    'boundary_stations': list(_get_boundary_stations(nodesDf, pd.DataFrame())),
    'boundary_events_added': False,
    'headway_layer_added': False,
})

# diagnostics
xmap = _station_x_map(G, nodesDf)
print('station_x_map contains COLD:', 'COLD' in xmap)
print('COLD pk_rel:', nodesDf.loc['COLD','pk_rel'])

# draw
draw_ean_plotly(G, nodesDf, fig, draw_edges=False, draw_nodes=True, is_scheduled=True)

print('Number of traces after drawing:', len(fig.data))
for i, t in enumerate(fig.data):
    print(i, getattr(t, 'name', None), t.mode, t.meta if hasattr(t, 'meta') else None)

# print boundary stations
print('boundary_stations meta:', fig.layout.meta.get('boundary_stations'))
