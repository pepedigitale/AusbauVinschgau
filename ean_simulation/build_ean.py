"""
Event-Activity Network (EAN) construction for a single-track railway line.

Design
------
Nodes  = timetable events: (train_id, station, 'arr'|'dep', seq_index)
Edges  = activities with a MINIMUM DURATION attribute `min_duration` (seconds):
    - "running"        : departure(station i) -> arrival(station i+1)
    - "dwell"          : arrival(station i)    -> departure(station i)
    - "schedule_floor" : SOURCE -> first departure of a train
                          (encodes "a train may not depart before the plan")
    - "headway"        : event of train i -> corresponding event of train j
                          on a shared resource (block/loop/platform), added
                          separately once you supply your computed headways

No dispatching / no reordering is assumed: train order at every shared
resource is fixed a priori (normally: scheduled order), so headway arcs are
directed from the earlier train's event to the later train's event and the
graph stays a DAG. Realized time of any event = max over all incoming arcs
of (realized_time(tail) + min_duration(arc)), evaluated in topological
order -> this is the propagation step used later for both the deterministic
sanity check below and, eventually, for each Monte Carlo draw.
"""

import numpy as np
import networkx as nx
from datetime import datetime


# --------------------------------------------------------------------------
# 1. Loading & time handling
# --------------------------------------------------------------------------

def load_trip_data(path: str) -> dict:
    """Load the .npy timetable dict: {train_id: [(station, arr_dt, dep_dt), ...]}."""
    return np.load(path, allow_pickle=True).item()


def to_seconds(dt: datetime) -> float:
    """
    Convert a timetable datetime to seconds-since-midnight.
    The dummy date (1900-1-1) is irrelevant; only time-of-day matters.
    NOTE: if any train run crosses midnight, extend this to carry a day
    offset instead of wrapping blindly -- flagged again below.
    """
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6


# --------------------------------------------------------------------------
# 2. Core EAN construction from trip_data alone
# --------------------------------------------------------------------------

def build_ean(trip_data: dict) -> nx.DiGraph:
    G = nx.DiGraph()

    for train_id, stops in trip_data.items():
        n = len(stops)
        arr_nodes = {}
        dep_nodes = {}

        for i, (station, arr_dt, dep_dt, is_stop) in enumerate(stops):
            arr_sec = to_seconds(arr_dt)
            dep_sec = to_seconds(dep_dt)

            # Arrival event exists for every stop except the train's origin
            if i > 0:
                arr_node = (train_id, station, "arr", i)
                G.add_node(
                    arr_node,
                    train=train_id,
                    station=station,
                    event="arr",
                    seq=i,
                    time=arr_sec,
                    scheduled_time=0,
                    is_stop=is_stop
                )
                arr_nodes[i] = arr_node

            # Departure event exists for every stop except the terminus
            if i < n - 1:
                dep_node = (train_id, station, "dep", i)

                attrs = dict(
                    train=train_id,
                    station=station,
                    event="dep",
                    seq=i,
                    time=dep_sec,
                    scheduled_time=0,
                    is_stop=is_stop
                )

                if is_stop:
                    attrs["scheduled_time"] = dep_sec

                G.add_node(dep_node, **attrs)
                dep_nodes[i] = dep_node

            # --- dwell arc: arrival -> departure at the same intermediate stop
            if i in arr_nodes and i in dep_nodes:
                dwell = dep_sec - arr_sec
                if dwell < 0:
                    dwell += 24 * 3600  # midnight-wrap guard, see note above
                min_dwell = 30 if is_stop else 0

                G.add_edge(
                    arr_nodes[i],
                    dep_nodes[i],
                    min_duration=min_dwell,
                    kind="dwell",
                    train=train_id,
                    station=station,
                )

            # --- running arc: previous departure -> this arrival
            if (i - 1) in dep_nodes and i in arr_nodes:
                prev_dep_sec = G.nodes[dep_nodes[i - 1]]["time"]
                run = arr_sec - prev_dep_sec
                if run < 0:
                    run += 24 * 3600
                G.add_edge(dep_nodes[i - 1], arr_nodes[i],
                           min_duration=run/1.07, kind="running", train=train_id,
                           from_station=G.nodes[dep_nodes[i - 1]]["station"],
                           to_station=station)

    return G


# --------------------------------------------------------------------------
# 3. Headway arc integration -- YOU adapt the input schema to your data
# --------------------------------------------------------------------------

def add_headway_arcs(G: nx.DiGraph, headway_constraints: list[dict]) -> nx.DiGraph:
    """
    Attach headway arcs to an existing EAN.

    Each element of `headway_constraints` is expected to describe ONE
    directed minimum-headway relation between two consecutive trains on a
    shared resource, e.g.:

        {
            "train_i": 1, "seq_i": 2, "event_i": "dep",   # earlier train
            "train_j": 3, "seq_j": 2, "event_j": "dep",   # following train
            "min_headway": 180.0,                          # seconds
            "resource": "MAR-block_section_4",             # free-text label
        }

    train_i's event is assumed to occur BEFORE train_j's event (this is
    where the "no reordering" assumption is enforced: you decide the order,
    e.g. by scheduled time, when you build this list from your conflict
    detector output -- the function itself does not reorder anything).

    This is intentionally a thin, format-agnostic adapter: reshape your
    minimum-headway results (whatever their current shape) into this list
    of dicts, and this function does the rest.
    """
    for hc in headway_constraints:
        u = (hc["train_i"], _station_lookup(G, hc["train_i"], hc["seq_i"]),
             hc["event_i"], hc["seq_i"])
        v = (hc["train_j"], _station_lookup(G, hc["train_j"], hc["seq_j"]),
             hc["event_j"], hc["seq_j"])

        if u not in G or v not in G:
            raise KeyError(f"Headway constraint references a node not in "
                            f"the EAN: {u} -> {v}")

        G.add_edge(u, v, min_duration=hc["min_headway"], kind="headway",
                   resource=hc.get("resource"))
    return G


def _station_lookup(G, train_id, seq):
    """Find the station name for a given (train, seq) pair already in G."""
    for _, data in G.nodes(data=True):
        if data.get("train") == train_id and data.get("seq") == seq:
            return data["station"]
    raise KeyError(f"No node found for train {train_id}, seq {seq}")


# --------------------------------------------------------------------------
# 4. Propagation (longest path / max-plus) -- also serves as a sanity check
# --------------------------------------------------------------------------

def propagate(G: nx.DiGraph, perturbations: dict | None = None) -> nx.DiGraph:
    """
    Propagate delays through an EAN.

    Parameters
    ----------
    G : nx.DiGraph
        Scheduled EAN. Nodes have a 'time' attribute and edges a
        'min_duration' attribute.

    perturbations : dict
        {(u, v): extra_seconds} to be added to edge min_duration.

    Returns
    -------
    nx.DiGraph
        A copy of G with node attribute 'time' updated to the realized times.
    """
    perturbations = perturbations or {}

    G_real = G.copy()

    realized = {}

    for node in nx.topological_sort(G_real):
        preds = list(G_real.predecessors(node))

        if not preds:
            realized[node] = G_real.nodes[node]["scheduled_time"]
            continue

        candidate = max(
            realized[p]
            + G_real.edges[p, node]["min_duration"]
            + perturbations.get((p, node), 0.0)
            for p in preds
        )

        realized[node] = max(candidate, G_real.nodes[node]["scheduled_time"])

    # Overwrite the time attribute
    nx.set_node_attributes(G_real, realized, "time")

    return G_real

def propagate2(
    G: nx.DiGraph,
    edge_perturbations: dict | None = None,
    node_perturbations: dict | None = None,
) -> nx.DiGraph:
    """
    Propagate delays through an EAN.

    Parameters
    ----------
    G : nx.DiGraph
        Scheduled EAN. Nodes have attributes including 'scheduled_time';
        edges have a 'min_duration' attribute.

    edge_perturbations : dict, optional
        {(u, v): extra_seconds} added to the corresponding edge duration.

    node_perturbations : dict, optional
        {node: extra_seconds} added to the realized time of that event.

    Returns
    -------
    nx.DiGraph
        A copy of G with node attribute 'time' updated to the realized times.
    """

    edge_perturbations = edge_perturbations or {}
    node_perturbations = node_perturbations or {}

    G_real = G.copy()
    realized = {}

    for node in nx.topological_sort(G_real):

        preds = list(G_real.predecessors(node))

        # Source events (no predecessors)
        if not preds:
            candidate = G_real.nodes[node]["scheduled_time"]
        else:
            candidate = max(
                realized[p]
                + G_real.edges[p, node]["min_duration"]
                + edge_perturbations.get((p, node), 0.0)
                for p in preds
            )

        # External delay at this event
        candidate += node_perturbations.get(node, 0.0)

        # No event may occur before its scheduled time
        realized[node] = max(candidate, G_real.nodes[node]["scheduled_time"])

    # Update realized times
    nx.set_node_attributes(G_real, realized, "time")

    return G_real

# --------------------------------------------------------------------------
# 5. Find out at what times which trains traverse chain boundaries (rank >= 3)
# --------------------------------------------------------------------------

from datetime import timedelta

def enrich_trip_data_with_boundaries(trip_data, routes, nodesDf, boundary_nodes):
    """
    Return a copy of trip_data where every train additionally contains
    interpolated timestamps at chain boundary nodes (rank >= 3).

    Existing timetable stops are left unchanged.
    Virtual boundary events are inserted in chronological order.

    Parameters
    ----------
    trip_data : dict
        {train_id: [(node, arr_time, dep_time), ...]}

    routes : dict
        {train_id: [ordered infrastructure nodes]}

    nodesDf : DataFrame
        Must contain column 'pk_rel'.

    chains : dict
        Output of enrich_trip_data_with_boundarieschains().

    Returns
    -------
    enriched_trip_data : dict
    """

    enriched = {}

    for train_id, route in routes.items():

        # Original timetable
        stops = list(trip_data[train_id])
        stop_names = [s[0] for s in stops]

        new_events = list(stops)

        for boundary in boundary_nodes:

            # Train does not traverse this boundary
            if boundary not in route:
                continue

            # Already a real timetable event
            if boundary in stop_names:
                continue

            boundary_idx = route.index(boundary)

            # Previous and next timetable stop along the route
            prev = None
            nxt = None

            for stop in stops:
                try:
                    idx = route.index(stop[0])
                except ValueError:
                    print(f"Train {train_id}")
                    print(f"Missing stop: {stop[0]}")
                    print(f"Route: {route}")
                    raise
                if idx < boundary_idx:
                    prev = stop

                if idx > boundary_idx:
                    nxt = stop
                    break

            if prev is None or nxt is None:
                continue

            prev_node, _, prev_dep, _ = prev
            next_node, next_arr, _, _ = nxt

            pk_prev = nodesDf.loc[prev_node, "pk_rel"]
            pk_next = nodesDf.loc[next_node, "pk_rel"]
            pk_boundary = nodesDf.loc[boundary, "pk_rel"]

            # Linear interpolation fraction
            frac = (pk_boundary - pk_prev) / (pk_next - pk_prev)

            run_time = next_arr - prev_dep

            boundary_time = prev_dep + frac * run_time

            # Virtual event:
            # arrival == departure because there is no stop
            new_events.append(
                (boundary, boundary_time, boundary_time, False)
            )

        # Keep chronological order
        new_events.sort(key=lambda x: x[1])

        enriched[train_id] = new_events

    return enriched