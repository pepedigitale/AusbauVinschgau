"""
Interactive Plotly visualization of an Event-Activity Network.

Layout:
    x = position along the line (pk_rel)
    y = time of day in seconds, descending from top to bottom.

The public functions intentionally have the same names as the Matplotlib
version:

    plot_ean(...)
    draw_ean(...)
"""

from collections import defaultdict

import numpy as np
import plotly.graph_objects as go


# ----------------------------------------------------------------------
# Edge styles
# ----------------------------------------------------------------------

EDGE_STYLE = {
    "running":        dict(color="#888888", width=1.2, dash="solid"),
    "dwell":          dict(color="#888888", width=1.2, dash="dot"),
    "schedule_floor": dict(color="#cccccc", width=0.8, dash="dash"),
    "headway_active": dict(color="#d62728", width=1.8, dash="solid"),
    "headway_inactive": dict(color="#888888", width=1.2, dash="solid"),
}

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _seconds_to_hhmmss(seconds):
    """Convert seconds since midnight to HH:MM:SS."""
    seconds = float(seconds)

    hours = int(seconds // 3600) % 24
    minutes = int((seconds % 3600) // 60)
    secs = int(round(seconds % 60))

    if secs == 60:
        secs = 0
        minutes += 1

    if minutes == 60:
        minutes = 0
        hours = (hours + 1) % 24

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _train_colors(trains):
    """Assign deterministic random colors to trains.

    Each train gets a consistent color derived from its identifier
    so the mapping is stable across runs but independent of the
    total number of trains or predefined palettes.
    """

    import random

    colors = {}

    for train in sorted(trains):
        rnd = random.Random(str(train))
        # avoid extremes (too dark or too light)
        r = rnd.randint(40, 215)
        g = rnd.randint(40, 215)
        b = rnd.randint(40, 215)
        colors[train] = f"#{r:02x}{g:02x}{b:02x}"

    return colors


def infer_train_colors(G):
    """
    Infer edge colors per train.

    Rules:
    - Fast trains -> blue
    - Slow trains -> red/green alternating by first-seen minute-group
    - Respect hourly color pattern per origin: for trains whose
      earliest event is at the same minute-of-hour from the same
      origin (e.g., "ME" or "MAL"), use the same color.

    The function returns a dict: train -> color (hex string).
    """

    # Collect per-train info: node count, earliest time, origin station
    trains = sorted({G.nodes[n]["train"] for n in G.nodes if n != "SOURCE"})

    train_info = {}

    for train in trains:
        nodes = [n for n in G.nodes if n != "SOURCE" and G.nodes[n]["train"] == train]
        if not nodes:
            continue

        times = [float(G.nodes[n]["time"]) for n in nodes]
        earliest_idx = int(np.argmin(times))
        earliest_time = times[earliest_idx]
        # find the node with earliest_time (first occurrence)
        earliest_nodes = [n for n in nodes if float(G.nodes[n]["time"]) == earliest_time]
        origin_node = earliest_nodes[0]
        origin_station = G.nodes[origin_node]["station"]

        unique_stations = {G.nodes[n]["station"] for n in nodes}
        node_count = len(unique_stations)

        minute = (int(earliest_time) // 60) % 60

        train_info[train] = dict(
            node_count=node_count,
            earliest_time=earliest_time,
            origin_station=origin_station,
            minute=minute,
        )

    if not train_info:
        return {}

    # Decide fast vs slow: use average (mean) node count as threshold
    node_counts = [info["node_count"] for info in train_info.values()]
    mean = float(np.mean(node_counts))

    for t, info in train_info.items():
        info["is_fast"] = info["node_count"] <= mean

    # Group trains by origin_station and minute
    by_origin = {}
    for t, info in train_info.items():
        origin = info["origin_station"]
        by_origin.setdefault(origin, {})
        by_origin[origin].setdefault(info["minute"], [])
        by_origin[origin][info["minute"]].append((info["earliest_time"], t))

    # Colors
    BLUE = "#1f77b4"  # fast
    RED = "#d62728"
    GREEN = "#2ca02c"

    result = {}

    # For each origin, iterate minute groups in chronological order of first appearance
    for origin, minutes in by_origin.items():
        # sort by earliest_time within each minute then by minute order of first occurrence
        minute_items = []
        for m, entries in minutes.items():
            first_time = min(time for time, _ in entries)
            minute_items.append((first_time, m))

        minute_items.sort()

        slow_toggle = True

        for _, m in minute_items:
            entries = sorted(minutes[m])
            # determine color from the first train seen at this minute
            _, first_train = entries[0]
            if train_info[first_train]["is_fast"]:
                color = BLUE
            else:
                color = RED if slow_toggle else GREEN
                slow_toggle = not slow_toggle

            # apply to all trains in this minute/origin group
            for _, t in entries:
                result[t] = color

    return result


def _get_boundary_stations(nodesDf, edgesDf):
    """
    Determine stations where the infrastructure changes between
    single/double track.

    This is the same logic used for the infrastructure schematic.
    """

    pk_values = sorted(nodesDf["pk_rel"].unique())

    if len(pk_values) < 2:
        return set()

    interval_count = defaultdict(int)

    for _, edge in edgesDf.iterrows():

        pk1 = nodesDf.loc[edge["node_from"], "pk_rel"]
        pk2 = nodesDf.loc[edge["node_to"], "pk_rel"]

        a, b = sorted((pk1, pk2))

        for left, right in zip(pk_values[:-1], pk_values[1:]):
            if left >= a and right <= b:
                interval_count[(left, right)] += 1

    intervals = list(zip(pk_values[:-1], pk_values[1:]))

    boundary_pks = set()

    for i in range(1, len(intervals)):

        left_interval = intervals[i - 1]
        right_interval = intervals[i]

        if (
            interval_count[left_interval]
            != interval_count[right_interval]
        ):
            boundary_pks.add(pk_values[i])

    return {
        station
        for station in nodesDf.index
        if nodesDf.loc[station, "pk_rel"] in boundary_pks
    }


def _is_boundary_event(G, node, boundary_stations):
    """
    Identify virtual/boundary events.

    Explicit attributes are checked first. If they are not available,
    the station location is used as fallback.
    """

    attrs = G.nodes[node]

    for key in (
        "boundary",
        "is_boundary",
        "virtual",
        "is_virtual",
    ):
        if key in attrs:
            value = attrs[key]

            if isinstance(value, (bool, np.bool_)):
                return bool(value)

    return attrs.get("station") in boundary_stations


def _station_x_map(G, nodesDf):
    """Return station -> pk_rel mapping for stations present in G."""

    events = [
        node
        for node in G.nodes
        if node != "SOURCE"
    ]

    stations = sorted(
        {
            G.nodes[node]["station"]
            for node in events
        },
        key=lambda station: nodesDf.loc[station, "pk_rel"],
    )

    return {
        station: nodesDf.loc[station, "pk_rel"]
        for station in stations
    }


# ----------------------------------------------------------------------
# Infrastructure schematic
# ----------------------------------------------------------------------

def _add_infrastructure_schematic(fig, nodesDf, edgesDf):
    """
    Draw the single/double-track infrastructure schematic below the
    x-axis, with enough separation from the station labels.
    """

    pk_values = sorted(nodesDf["pk_rel"].unique())

    if len(pk_values) < 2:
        return

    interval_count = defaultdict(int)

    for _, edge in edgesDf.iterrows():

        pk1 = nodesDf.loc[edge["node_from"], "pk_rel"]
        pk2 = nodesDf.loc[edge["node_to"], "pk_rel"]

        a, b = sorted((pk1, pk2))

        for left, right in zip(pk_values[:-1], pk_values[1:]):
            if left >= a and right <= b:
                interval_count[(left, right)] += 1

    # Further below the x-axis than before.
    y = -0.13
    offset = 0.008

    for (left, right), n_tracks in sorted(interval_count.items()):

        if n_tracks == 1:

            fig.add_shape(
                type="line",
                xref="x",
                yref="paper",
                x0=left,
                x1=right,
                y0=y,
                y1=y,
                line=dict(
                    color="black",
                    width=2,
                ),
                layer="above",
            )

        else:

            for yy in (y - offset, y + offset):

                fig.add_shape(
                    type="line",
                    xref="x",
                    yref="paper",
                    x0=left,
                    x1=right,
                    y0=yy,
                    y1=yy,
                    line=dict(
                        color="black",
                        width=2,
                    ),
                    layer="above",
                )


# ----------------------------------------------------------------------
# Axes
# ----------------------------------------------------------------------

def _configure_axes(fig, nodesDf, title):
    """Configure x/y axes and overall layout."""

    # --------------------------------------------------------------
    # X axis
    # --------------------------------------------------------------

    pk_groups = defaultdict(list)

    lds_nodes = nodesDf.index[
        nodesDf["node_type"] == "LdS"
    ]

    for node in lds_nodes:
        pk_groups[nodesDf.loc[node, "pk_rel"]].append(node)

    xticks = []
    xticklabels = []

    for pk in sorted(pk_groups):

        xticks.append(pk)

        # Station names remain on the x axis.
        # Multiple names at the same pk are stacked.
        xticklabels.append(
            "<br>".join(pk_groups[pk]) + f"<br>({pk})"
        )

    fig.update_xaxes(
        tickmode="array",
        tickvals=xticks,
        ticktext=xticklabels,
        showgrid=True,
        gridcolor="lightgrey",
        gridwidth=0.6,
        zeroline=False,
        ticklabelstandoff=10,
    )

    # --------------------------------------------------------------
    # Y axis
    # --------------------------------------------------------------

    times = []

    for trace in fig.data:

        if trace.y is None:
            continue

        for value in trace.y:

            if value is not None:
                times.append(float(value))

    if times:

        y_min = min(times)
        y_max = max(times)

        # Same 15-minute grid as the Matplotlib implementation.
        tick_interval = 15 * 60

        tick_start = (
            tick_interval
            * np.floor(y_min / tick_interval)
        )

        tick_end = (
            tick_interval
            * np.ceil(y_max / tick_interval)
        )

        tickvals = np.arange(
            tick_start,
            tick_end + tick_interval,
            tick_interval,
        )

        ticktext = [
            _seconds_to_hhmmss(value)
            for value in tickvals
        ]

        fig.update_yaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,

            # Time increases downward.
            autorange="reversed",

            title_text="Time",
            showgrid=True,
            gridcolor="lightgrey",
            gridwidth=0.6,
            zeroline=False,
        )

    # --------------------------------------------------------------
    # General layout
    # --------------------------------------------------------------

    fig.update_layout(
        showlegend=False,
        title=dict(text=title),

        template="plotly_white",

        hovermode="closest",

        margin=dict(
            l=80,
            r=220,
            t=80,
            b=180,
        ),

        height=900,

        legend=dict(
            title=dict(text="Layers"),

            # Clicking a layer toggles the whole group.
            groupclick="togglegroup",

            itemsizing="constant",

            traceorder="grouped",

            x=1.02,
            xanchor="left",
            y=1,
            yanchor="top",
        ),
    )


# ----------------------------------------------------------------------
# Main plot function
# ----------------------------------------------------------------------

def plot_ean_plotly(
    G,
    nodesDf,
    edgesDf,
    title="Event-Activity Network",
    figsize=(20, 20),
):
    """
    Create a new Plotly figure and plot the scheduled EAN.

    The return signature intentionally matches the Matplotlib version:

        fig, ax = plot_ean(...)

    Here both returned values are the Plotly Figure.
    """

    fig = go.Figure()

    # Information shared by subsequent draw_ean() calls.
    fig.update_layout(
        meta=dict(
            realization_count=0,
            boundary_stations=list(
                _get_boundary_stations(
                    nodesDf,
                    edgesDf,
                )
            ),
            boundary_events_added=False,
            headway_layer_added=False,
        )
    )

    # Draw scheduled EAN.
    draw_ean_plotly(
        G,
        nodesDf,
        fig,
        alpha=1.0,
        linewidth_scale=1.0,
        layer_name="Scheduled",
        is_scheduled=True,
    )

    # Infrastructure schematic.
    _add_infrastructure_schematic(
        fig,
        nodesDf,
        edgesDf,
    )

    # Axes/layout.
    _configure_axes(
        fig,
        nodesDf,
        title,
    )

    return fig, fig


# ----------------------------------------------------------------------
# Draw/add EAN
# ----------------------------------------------------------------------

def draw_ean_plotly(
    G,
    nodesDf,
    ax,
    alpha=1.0,
    linewidth_scale=1.0,
    linestyle_override=None,
    draw_nodes=True,
    draw_edges=True,
    layer_name=None,
    is_scheduled=False,
):
    """
    Add an EAN to an existing Plotly Figure.

    Parameters are intentionally compatible with the Matplotlib
    implementation.

    Additional parameters:

    layer_name : str or None
        Name displayed in the Plotly legend.

    is_scheduled : bool
        Internal flag used by plot_ean().
    """

    fig = ax

    events = [
        node
        for node in G.nodes
        if node != "SOURCE"
    ]

    x_of_station = _station_x_map(
        G,
        nodesDf,
    )

    trains = sorted(
        {
            G.nodes[node]["train"]
            for node in events
        }
    )

    meta = fig.layout.meta or {}

    # Node/marker colors
    stored_node_colors = meta.get("node_colors")
    if is_scheduled or not stored_node_colors:
        color_of_train = _train_colors(trains)
        if is_scheduled:
            meta["node_colors"] = color_of_train
            fig.layout.meta = meta
    else:
        # reuse stored mapping, but ensure any newly appearing trains get a color
        color_of_train = dict(stored_node_colors)
        missing = set(trains) - set(color_of_train.keys())
        if missing:
            color_of_train.update(_train_colors(sorted(missing)))

    # Edge colors (inferred style per train)
    stored_edge_colors = meta.get("edge_train_colors")
    if is_scheduled or not stored_edge_colors:
        edge_color_of_train = infer_train_colors(G)
        if is_scheduled:
            meta["edge_train_colors"] = edge_color_of_train
            fig.layout.meta = meta
    else:
        edge_color_of_train = dict(stored_edge_colors)
        missing = set(trains) - set(edge_color_of_train.keys())
        if missing:
            # fallback: prefer node color for missing entries, else generate
            for t in missing:
                edge_color_of_train[t] = (
                    color_of_train.get(t)
                    or _train_colors([t])[t]
                )

    boundary_stations = set(
        fig.layout.meta.get(
            "boundary_stations",
            [],
        )
    )

    # ==============================================================
    # Determine layer name
    # ==============================================================

    if is_scheduled:

        layer_name = layer_name or "Scheduled"
        layer_id = "scheduled"

    else:

        count = int(
            fig.layout.meta.get("realization_count", 0)
        ) + 1

        fig.layout.meta["realization_count"] = count

        layer_name = (
            layer_name
            or f"Realization {count}"
        )

        layer_id = f"realization_{count}"

    legend_group=layer_id

    graph_meta = dict(
        layer_id=layer_id,
        layer_name=layer_name,
        layer_type="graph",
    )

    nodes_meta = dict(
        layer_id=layer_id,
        layer_name=layer_name,
        layer_type="nodes",
    )

    boundary_meta = dict(
        layer_id=layer_id,
        layer_name=layer_name,
        layer_type="boundary",
    )

    headway_meta = dict(
        layer_id=layer_id,
        layer_name=layer_name,
        layer_type="headway",
    )

    # ==============================================================
    # Proxy trace for the EAN layer
    #
    # This is what appears in the legend. All traces belonging to
    # this graph share the same legendgroup.
    # ==============================================================

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name=layer_name,
            legendgroup=legend_group,
            line=dict(
                color="#555555",
                width=3,
            ),
            showlegend=True,
            hoverinfo="skip",
        )
    )

    # ==============================================================
    # Edges
    # ==============================================================

    if draw_edges:

        for u, v, data in G.edges(data=True):

            kind = data.get(
                "kind",
                "running",
            )

            x0 = x_of_station[
                G.nodes[u]["station"]
            ]

            x1 = x_of_station[
                G.nodes[v]["station"]
            ]

            y0 = G.nodes[u]["time"]
            y1 = G.nodes[v]["time"]

            # ------------------------------------------------------
            # Headway edges
            # ------------------------------------------------------

            if kind == "headway":

                is_active = bool(data.get("is_active", False))

                if not is_active:
                    continue
                
                style_name = "headway_active" if is_active else "headway_inactive"
                style = EDGE_STYLE[style_name].copy()

                min_headway = data.get("min_duration")
                min_headway_text = (
                    _seconds_to_hhmmss(min_headway)
                    if min_headway is not None
                    else "n/a"
                )

                hover = (
                    "<b>Headway constraint</b>"
                    f"<br>Location: {G.nodes[u]['station']}"
                    f"<br>Start: {_seconds_to_hhmmss(y0)}"
                    f"<br>End: {_seconds_to_hhmmss(y1)}"
                    f"<br>Δt: "
                    f"{_seconds_to_hhmmss(y1 - y0)}"
                    f"<br>Min headway: {min_headway_text}"
                    f"<br>Status: {'active' if is_active else 'inactive'}"
                    "<extra></extra>"
                )

                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode="lines",
                        line=dict(
                            color=style["color"],
                            width=style["width"],
                            dash=style["dash"],
                        ),
                        opacity=alpha,
                        showlegend=False,
                        meta=headway_meta,
                        hovertemplate=hover,
                    )
                )

                continue

            # ------------------------------------------------------
            # Normal EAN edges
            # ------------------------------------------------------

            style = EDGE_STYLE.get(
                kind,
                dict(
                    color="black",
                    width=1,
                    dash="solid",
                ),
            ).copy()

            style["width"] *= linewidth_scale

            # For normal graph edges, color by train according to
            # the inferred per-train edge color mapping.
            train = G.nodes[u].get("train")
            if train is not None:
                style["color"] = edge_color_of_train.get(
                    train,
                    style.get("color", "black"),
                )

            if linestyle_override is not None:

                mpl_to_plotly_dash = {
                    "-": "solid",
                    "--": "dash",
                    ":": "dot",
                    "-.": "dashdot",
                }

                style["dash"] = mpl_to_plotly_dash.get(
                    linestyle_override,
                    linestyle_override,
                )

            hover = (
                f"<b>{layer_name}</b>"
                f"<br>Train: {train}"
                f"<br>Activity: {kind}"
                f"<br>{G.nodes[u]['station']} → "
                f"{G.nodes[v]['station']}"
                f"<br>Start: {_seconds_to_hhmmss(y0)}"
                f"<br>End: {_seconds_to_hhmmss(y1)}"
                f"<br>Duration: "
                f"{_seconds_to_hhmmss(abs(y1 - y0))}"
                "<extra></extra>"
            )

            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(
                        color=style["color"],
                        width=style["width"],
                        dash=style["dash"],
                    ),
                    opacity=alpha,
                    showlegend=False,
                    meta=graph_meta,
                    hovertemplate=hover,
                )
            )

    # ==============================================================
    # Nodes
    # ==============================================================

    if draw_nodes:

        normal_nodes = []
        boundary_nodes = []

        for node in events:

            if _is_boundary_event(
                G,
                node,
                boundary_stations,
            ):
                boundary_nodes.append(node)
            else:
                normal_nodes.append(node)

        # ----------------------------------------------------------
        # Normal events
        # ----------------------------------------------------------

        for train in trains:

            train_nodes = [
                node
                for node in normal_nodes
                if G.nodes[node]["train"] == train
            ]

            if not train_nodes:
                continue

            xs = [
                x_of_station[
                    G.nodes[node]["station"]
                ]
                for node in train_nodes
            ]

            ys = [
                G.nodes[node]["time"]
                for node in train_nodes
            ]

            symbols = [
                "circle"
                if G.nodes[node]["event"] == "dep"
                else "square"
                for node in train_nodes
            ]

            # Everything needed for hover is put into customdata.
            customdata = [
                [
                    train,
                    G.nodes[node]["station"],
                    G.nodes[node]["event"],
                    x_of_station[
                        G.nodes[node]["station"]
                    ],
                    _seconds_to_hhmmss(
                        G.nodes[node]["time"]
                    ),
                    G.nodes[node]["seq"],
                ]
                for node in train_nodes
            ]

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,

                    mode="markers",

                    marker=dict(
                        size=9,
                        symbol=symbols,
                        color=color_of_train[train],
                        line=dict(
                            color="black",
                            width=0.5,
                        ),
                    ),

                    opacity=alpha,

                    legendgroup=legend_group,

                    showlegend=False,

                    customdata=customdata,
                    meta=nodes_meta,
                    hovertemplate=(
                        f"<b>{layer_name}</b>"
                        "<br>Train: %{customdata[0]}"
                        "<br>Location: %{customdata[1]}"
                        "<br>Event: %{customdata[2]}"
                        "<br>Time: %{customdata[4]}"
                        "<br>pk: %{customdata[3]:.3f}"
                        "<br>Sequence: %{customdata[5]}"
                        "<extra></extra>"
                    ),
                )
            )

        # ----------------------------------------------------------
        # Boundary / virtual events
        # ----------------------------------------------------------

        if boundary_nodes:

            xs = [
                x_of_station[
                    G.nodes[node]["station"]
                ]
                for node in boundary_nodes
            ]

            ys = [
                G.nodes[node]["time"]
                for node in boundary_nodes
            ]

            customdata = [
                [
                    G.nodes[node]["train"],
                    G.nodes[node]["station"],
                    G.nodes[node].get(
                        "event",
                        "boundary",
                    ),
                    x_of_station[
                        G.nodes[node]["station"]
                    ],
                    _seconds_to_hhmmss(
                        G.nodes[node]["time"]
                    ),
                    G.nodes[node]["seq"],
                ]
                for node in boundary_nodes
            ]

            first_boundary_layer = not fig.layout.meta.get(
                "boundary_events_added",
                False,
            )

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,

                    mode="markers",

                    marker=dict(
                        size=11,
                        symbol="diamond-open",
                        color="black",
                        line=dict(
                            color="black",
                            width=1.5,
                        ),
                    ),

                    opacity=alpha,

                    legendgroup="boundary_events",

                    showlegend=first_boundary_layer,

                    name="Boundary / virtual events",

                    customdata=customdata,
                    meta=boundary_meta,
                    hovertemplate=(
                        f"<b>{layer_name}</b>"
                        "<br><b>Boundary / virtual event</b>"
                        "<br>Train: %{customdata[0]}"
                        "<br>Location: %{customdata[1]}"
                        "<br>Time: %{customdata[4]}"
                        "<br>pk: %{customdata[3]:.3f}"
                        "<br>Sequence: %{customdata[5]}"
                        "<extra></extra>"
                    ),
                )
            )

            fig.layout.meta["boundary_events_added"] = True



def show_ean(fig, filename="ean_plot.html", auto_open=True):
    """
    Save the Plotly EAN as a self-contained HTML file with a custom
    hierarchical layer control panel.

    The hierarchy is:

        Scheduled
            Graph
            Boundary events
            Headways

        Realization 1
            Graph
            Boundary events
            Headways
    """

    import uuid
    from pathlib import Path

    plot_id = "ean_plot_" + uuid.uuid4().hex

    config = {
        "scrollZoom": True,
        "displayModeBar": True,
        "responsive": True,
        "doubleClick": "reset",
    }

    html = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id=plot_id,
        config=config,
    )

    fig.update_layout(showlegend=False)

    html = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id=plot_id,
        config=config,
    )

    wrapper = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">

<style>

body {{
    margin: 0;
    font-family: Arial, sans-serif;
    background: white;
}}

#ean-container {{
    display: flex;
    width: 100vw;
    height: 100vh;
    overflow: visible;
}}

#plot-container {{
    flex: 1;
    min-width: 0;
    height: 100%;
}}

#layer-panel {{
    width: 230px;
    padding: 16px 14px;
    border-left: 1px solid #cccccc;
    background: #fafafa;
    overflow-y: auto;
    box-sizing: border-box;
}}

#layer-panel h3 {{
    margin: 0 0 14px 0;
    font-size: 16px;
}}

.layer-row {{
    display: flex;
    align-items: center;
    margin: 5px 0;
    white-space: nowrap;
}}

.layer-row.parent {{
    margin-top: 12px;
    font-weight: 600;
}}

.layer-row.child {{
    margin-left: 24px;
    font-weight: normal;
    font-size: 13px;
}}

.layer-row input {{
    margin-right: 7px;
}}

</style>
</head>

<body>

<div id="ean-container">

    <div id="plot-container">
        {html}
    </div>

    <div id="layer-panel">
        <h3>Layers</h3>
        <div id="layer-controls"></div>
    </div>

</div>


<script>

(function() {{

    const gd = document.getElementById("{plot_id}");
    const controls = document.getElementById("layer-controls");

    /*
     * Each Plotly trace gets metadata like:
     *
     * meta: {{
     *     layer_id: "realization_1",
     *     layer_type: "graph"
     * }}
     *
     * We use this to construct the hierarchy.
     */

    const layers = {{}};

    gd.data.forEach((trace, index) => {{

        if (!trace.meta || !trace.meta.layer_id) {{
            return;
        }}

        const layerId = trace.meta.layer_id;
        const layerType = trace.meta.layer_type;

        if (!layers[layerId]) {{
            layers[layerId] = {{
                name: trace.meta.layer_name,
                traces: {{}}
            }};
        }}

        if (!layers[layerId].traces[layerType]) {{
            layers[layerId].traces[layerType] = [];
        }}

        layers[layerId].traces[layerType].push(index);
    }});


    const typeNames = {{
        graph: "Graph",
        nodes: "Nodes",
        boundary: "Boundary events",
        headway: "Headways"
    }};


    /*
     * Visibility state.
     *
     * This is kept separately from Plotly's trace visibility so that
     * the parent checkbox can operate as a master switch without
     * destroying the state of the individual children.
     */

    const state = {{}};

    Object.keys(layers).forEach(layerId => {{

        const isScheduled = layerId === "scheduled";

        state[layerId] = {{
            master: isScheduled,
            graph: isScheduled,
            nodes: false,
            boundary: false,
            headway: isScheduled
        }};

    }});


    function tracesFor(layerId, type) {{
        return layers[layerId].traces[type] || [];
    }}


    function setTraces(indices, visible) {{

        if (!indices.length) {{
            return;
        }}

        Plotly.restyle(
            gd,
            {{visible: visible}},
            indices
        );
    }}


    function effectiveVisibility(layerId, type) {{

        return (
            state[layerId].master &&
            state[layerId][type]
        );

    }}


    function updateLayer(layerId) {{

        ["graph", "nodes", "boundary", "headway"].forEach(type => {{

            const indices = tracesFor(layerId, type);

            setTraces(
                indices,
                effectiveVisibility(layerId, type)
            );

        }});

        updateParentCheckbox(layerId);
        updateChildCheckboxes(layerId);
    }}


    function updateParentCheckbox(layerId) {{

        const layer = layers[layerId];

        const parent = layer.parentCheckbox;

        if (!parent) {{
            return;
        }}

        const children = [
            state[layerId].graph,
            state[layerId].nodes,
            state[layerId].boundary,
            state[layerId].headway
        ];

        const allOn = children.every(x => x);
        const allOff = children.every(x => !x);

        if (!state[layerId].master) {{

            parent.checked = false;
            parent.indeterminate = false;

        }} else if (allOn) {{

            parent.checked = true;
            parent.indeterminate = false;

        }} else if (allOff) {{

            parent.checked = false;
            parent.indeterminate = false;

        }} else {{

            parent.checked = true;
            parent.indeterminate = true;

        }}
    }}


    function updateChildCheckboxes(layerId) {{

        const layer = layers[layerId];

        if (!layer || !layer.childCheckboxes) {{
            return;
        }}

        ["graph", "nodes", "boundary", "headway"].forEach(type => {{
            const checkbox = layer.childCheckboxes[type];
            if (checkbox) {{
                checkbox.checked = state[layerId][type];
            }}
        }});
    }}


    function makeCheckbox(checked) {{

        const checkbox = document.createElement("input");

        checkbox.type = "checkbox";
        checkbox.checked = checked;

        return checkbox;
    }}


    Object.entries(layers).forEach(
        ([layerId, layer]) => {{

        /*
         * ----------------------------------------------------------
         * Parent row
         * ----------------------------------------------------------
         */

        const parentRow =
            document.createElement("div");

        parentRow.className =
            "layer-row parent";

        const parentCheckbox =
            makeCheckbox(state[layerId].master);

        const parentLabel =
            document.createElement("span");

        parentLabel.textContent =
            layer.name;

        parentRow.appendChild(parentCheckbox);
        parentRow.appendChild(parentLabel);

        controls.appendChild(parentRow);

        layer.parentCheckbox =
            parentCheckbox;

        layer.childCheckboxes = {{}};


        parentCheckbox.addEventListener(
            "change",
            function() {{

                state[layerId].master =
                    this.checked;

                ["graph", "nodes", "boundary", "headway"].forEach(
                    type => {{
                        state[layerId][type] = this.checked;
                        const child = layer.childCheckboxes[type];
                        if (child) {{
                            child.checked = this.checked;
                        }}
                    }}
                );

                updateLayer(layerId);

            }}
        );


        /*
         * ----------------------------------------------------------
         * Child rows
         * ----------------------------------------------------------
         */

        ["graph", "nodes", "boundary", "headway"].forEach(
            type => {{

            const row =
                document.createElement("div");

            row.className =
                "layer-row child";

            const checkbox =
                makeCheckbox(state[layerId][type]);

            layer.childCheckboxes[type] = checkbox;

            const label =
                document.createElement("span");

            label.textContent =
                typeNames[type];

            row.appendChild(checkbox);
            row.appendChild(label);

            controls.appendChild(row);


            checkbox.addEventListener(
                "change",
                function() {{

                    state[layerId][type] =
                        this.checked;

                    updateLayer(layerId);

                }}
            );

        }});

    }});


    /*
     * Initial visibility.
     */

    Object.keys(layers).forEach(
        layerId => updateLayer(layerId)
    );

}})();

</script>

</body>
</html>
"""

    path = Path(filename)
    path.write_text(
        wrapper,
        encoding="utf-8",
    )

    print(f"EAN visualization written to: {path.resolve()}")

    if auto_open:
        import webbrowser
        webbrowser.open(path.resolve().as_uri())

    return path