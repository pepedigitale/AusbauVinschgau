"""
Visualize an Event-Activity Network built by build_ean.py.

Layout: x = scheduled (or realized) time-of-day, y = location along the
line (from nodesDf.pk_rel if available, otherwise order of first
appearance in the graph). Events colored by train, arcs styled by kind.
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.cm import get_cmap
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib.transforms as transforms



EDGE_STYLE = {
    "running":        dict(color="#888888", linewidth=1.2, linestyle="-",   zorder=1),
    "dwell":          dict(color="#888888", linewidth=1.2, linestyle=":",   zorder=1),
    "schedule_floor": dict(color="#cccccc", linewidth=0.8, linestyle="--",  zorder=0),
    "headway":        dict(color="#d62728", linewidth=1.8, linestyle="-",   zorder=2),
}


def _seconds_to_hhmm(s):
    h = int(s // 3600) % 24
    m = int((s % 3600) // 60)
    return f"{h:02d}:{m:02d}"


def plot_ean(G,
             nodesDf,
             edgesDf,
             title="Event-Activity Network",
             figsize=(20,20)):
    """
    Create a new figure and plot a single EAN.
    """
    from collections import defaultdict

    pk_to_nodes = defaultdict(list)

    for node in nodesDf.index:
        pk_to_nodes[nodesDf.loc[node, "pk_rel"]].append(node)

    ambiguous_nodes = {
        n
        for nodes in pk_to_nodes.values()
        if len(nodes) > 1
        for n in nodes
    }

    fig, ax = plt.subplots(figsize=figsize)

    kinds_present, color_of_train = draw_ean(
        G,
        nodesDf,
        ax
    )

    events = [n for n in G.nodes if n != "SOURCE"]

    for n in events:

        station = G.nodes[n]["station"]

        if station not in ambiguous_nodes:
            continue

        ax.text(
            nodesDf.loc[station, "pk_rel"] + 0.08,
            G.nodes[n]["time"],
            station,
            fontsize=7,
            va="center",
            color="dimgray",
        )

    pk_groups = defaultdict(list)

    lds_nodes = nodesDf.index[nodesDf["node_type"] == "LdS"]

    for node in lds_nodes:
        pk_groups[nodesDf.loc[node, "pk_rel"]].append(node)

    xticks = []
    xticklabels = []

    for pk in sorted(pk_groups):

        xticks.append(pk)

        xticklabels.append(
            "\n".join(pk_groups[pk]) + f"\n({pk})"
        )

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([_seconds_to_hhmm(t) for t in yticks])

    ax.invert_yaxis()

    ax.grid(True, color="lightgrey", linewidth=0.6)

    ax.set_title(title)

    trains = sorted({G.nodes[n]["train"] for n in events})

    train_handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=color_of_train[t],
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=f"train {t}",
        )
        for t in trains
    ]

    kind_handles = [
        mlines.Line2D(
            [],
            [],
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            label=kind,
        )
        for kind, style in EDGE_STYLE.items()
        if kind in kinds_present
    ]

    marker_handles = [
        mlines.Line2D([], [], marker="o", linestyle="", color="grey",
                      label="departure"),
        mlines.Line2D([], [], marker="s", linestyle="", color="grey",
                      label="arrival"),
    ]

    leg1 = ax.legend(
        handles=train_handles,
        title="Train",
        loc="upper left",
        bbox_to_anchor=(1.01,1),
        fontsize=8,
    )

    ax.add_artist(leg1)

    ax.legend(
        handles=kind_handles + marker_handles,
        title="Arc kind / event",
        loc="lower left",
        bbox_to_anchor=(1.01,0),
        fontsize=8,
    )

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: _seconds_to_hhmm(x))
    )

    trans = transforms.blended_transform_factory(
        ax.transData,
        ax.transAxes,
    )

    # ------------------------------------------------------------------
    # Infrastructure schematic (single/double track)
    # ------------------------------------------------------------------

    # elementary pk intervals
    pk_values = sorted(nodesDf["pk_rel"].unique())

    # count how many infrastructure edges cover each interval
    interval_count = defaultdict(int)

    for _, edge in edgesDf.iterrows():

        pk1 = nodesDf.loc[edge["node_from"], "pk_rel"]
        pk2 = nodesDf.loc[edge["node_to"], "pk_rel"]

        a, b = sorted((pk1, pk2))

        for left, right in zip(pk_values[:-1], pk_values[1:]):
            if left >= a and right <= b:
                interval_count[(left, right)] += 1

    # x in data coordinates, y in axes coordinates
    trans = transforms.blended_transform_factory(
        ax.transData,
        ax.transAxes,
    )

    y = -0.015

    for (left, right), n_tracks in sorted(interval_count.items()):

        if n_tracks == 1:
            ax.plot(
                [left, right],
                [y, y],
                transform=trans,
                color="black",
                linewidth=1.2,
                solid_capstyle="butt",
                clip_on=False,
                zorder=10,
            )
        else:
            offset = 0.002

            for yy in (y - offset, y + offset):
                ax.plot(
                    [left, right],
                    [yy, yy],
                    transform=trans,
                    color="black",
                    linewidth=1.2,
                    solid_capstyle="butt",
                    clip_on=False,
                    zorder=10,
                )

    ax.tick_params(axis="x", pad=25)

    fig.tight_layout()

    return fig, ax

def draw_ean(G, nodesDf, ax,
             alpha=1.0,
             linewidth_scale=1.0,
             linestyle_override=None,
             draw_nodes=True,
             draw_edges=True):
    """
    Draw an Event-Activity Network onto an existing matplotlib Axes.

    Parameters
    ----------
    G : nx.DiGraph
    nodesDf : DataFrame
        Must contain the pk_rel column.
    ax : matplotlib.axes.Axes
    alpha : float
    linewidth_scale : float
    linestyle_override : str or None
        e.g. "--" to force dashed lines.
    draw_nodes : bool
    draw_edges : bool
    """

    events = [n for n in G.nodes if n != "SOURCE"]

    stations = sorted(
        {G.nodes[n]["station"] for n in events},
        key=lambda s: nodesDf.loc[s, "pk_rel"]
    )

    x_of_station = {
        s: nodesDf.loc[s, "pk_rel"]
        for s in stations
    }

    def y_of(node):
        return G.nodes[node]["time"]

    trains = sorted({G.nodes[n]["train"] for n in events})
    cmap = get_cmap("tab10" if len(trains) <= 10 else "tab20")
    color_of_train = {t: cmap(i % cmap.N) for i, t in enumerate(trains)}

    kinds_present = set()

    # ---------------- edges ----------------

    if draw_edges:
        for u, v, data in G.edges(data=True):

            if u == "SOURCE":
                continue

            kind = data.get("kind", "running")
            kinds_present.add(kind)

            style = EDGE_STYLE.get(
                kind,
                dict(color="black", linewidth=1, linestyle="-")
            ).copy()

            style["alpha"] = alpha
            style["linewidth"] *= linewidth_scale

            if linestyle_override is not None:
                style["linestyle"] = linestyle_override

            ax.plot(
                [x_of_station[G.nodes[u]["station"]],
                 x_of_station[G.nodes[v]["station"]]],
                [y_of(u),
                 y_of(v)],
                **style
            )

    # ---------------- nodes ----------------

    if draw_nodes:

        for n in events:

            marker = "o" if G.nodes[n]["event"] == "dep" else "s"

            ax.scatter(
                x_of_station[G.nodes[n]["station"]],
                y_of(n),
                s=35,
                marker=marker,
                color=color_of_train[G.nodes[n]["train"]],
                edgecolor="black",
                linewidth=0.4,
                alpha=alpha,
                zorder=3,
            )

    times = [
    G.nodes[n]["time"]
    for n in G.nodes
    if n != "SOURCE"
    ]

    ymin, ymax = sorted(ax.get_ylim())

    ticks_min = 15
    tk = ticks_min * 60

    tick_start = tk * np.floor(ymin / tk)
    tick_end   = tk * np.ceil(ymax / tk)

    ax.set_yticks(np.arange(tick_start, tick_end + tk, tk))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: _seconds_to_hhmm(x)))

    return kinds_present, color_of_train