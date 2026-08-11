"""
Turns (nodesDf, edgesDf, routes, your headway dict) into headway arcs on the
EAN built by build_ean.py.

Pipeline
--------
1. build_infra_graph      : nodesDf/edgesDf -> undirected infra graph
2. extract_chains         : split graph at degree>=3 (junctions) and
                             degree==1 (line ends) nodes into 1-D chains,
                             matching your own "rank>=3" segmentation
3. train_direction        : 'up'/'down' from pk_rel monotonicity along a
                             train's route
4. train_speed_category   : 'fast'/'slow' heuristic from stop pattern vs.
                             nodesDf.stop_fast / stop_slow -- FLAGGED, verify
5. chain_boundary_event    : maps a chain's physical boundary node to the
                             nearest real EAN event in a train's route
6. assemble_headway_constraints : for every chain, order the trains that use
                             it by scheduled entry time, and emit one
                             headway constraint per consecutive pair, using
                             the correct (cat_i, cat_j) entry from your
                             16-value-per-chain dict
7. add_headway_arcs (from build_ean.py) attaches them to G

ASSUMPTIONS TO VERIFY BEFORE TRUSTING THE OUTPUT
-------------------------------------------------
- UP_IS_INCREASING_PK : whether increasing pk_rel along a route means 'up'
  or 'down' in your headway-dict labels. Check one train you know the
  direction of and flip the flag below if needed.
- train_speed_category(): heuristic only. Spot-check a couple of known
  fast/slow trains before trusting the classification for all of them.
"""

import networkx as nx
import pandas as pd
import numpy as np

UP_IS_INCREASING_PK = True  # <-- VERIFY against a known train, see docstring


# --------------------------------------------------------------------------
# 1-2. Infra graph & chain extraction
# --------------------------------------------------------------------------

def build_infra_graph(edgesDf: pd.DataFrame) -> nx.Graph:
    IG = nx.Graph()
    for edge_id, row in edgesDf.iterrows():
        IG.add_edge(row["node_from"], row["node_to"],
                    length=row["length"], edge_id=edge_id)
    return IG

def extract_chains2(IG: nx.Graph, nodesDf: pd.DataFrame):
    """
    Returns
        chains: {(start_boundary, end_boundary): [path]}
        boundary_nodes: set

    A chain is a simple path between two boundary nodes (degree >= 3 or degree == 1).
    Chains are stored in the direction in which they are discovered.
    If multiple physical paths connect the same pair of boundaries,
    they are all stored.
    """

    from collections import defaultdict

    boundary_nodes = {
        n for n, d in IG.degree()
        if d >= 3 or d == 1
    }

    chains = defaultdict(list)

    for start in boundary_nodes:

        for nbr in IG.neighbors(start):

            path = [start, nbr]
            prev, curr = start, nbr

            while curr not in boundary_nodes:

                nxts = [x for x in IG.neighbors(curr) if x != prev]

                if len(nxts) == 0:
                    break

                if len(nxts) > 1:
                    raise RuntimeError(
                        f"Unexpected branch inside chain at {curr}: {nxts}"
                    )

                prev, curr = curr, nxts[0]
                path.append(curr)

            if curr in boundary_nodes:
                chains[(start, curr)].append(path)

    return dict(chains), boundary_nodes
    

def extract_chains3(IG: nx.Graph, nodesDf: pd.DataFrame):
    """
    Returns
        chains: {(start_boundary, end_boundary): [ordered list of nodes]}
        boundary_nodes: set

    A chain is a simple path between two adjacent boundary nodes
    (boundary = degree >= 3 or degree == 1).

    Keys are directional: (A,B) and (B,A) are different keys only if they
    correspond to different physical corridors. Reversing an already
    discovered chain is NOT added.
    """

    boundary_nodes = {
        n for n, d in IG.degree()
        if d >= 3 or d == 1
    }

    chains = {}
    visited_edges = set()

    for start in boundary_nodes:

        for nbr in IG.neighbors(start):

            edge = frozenset((start, nbr))
            if edge in visited_edges:
                continue

            path = [start, nbr]
            visited_edges.add(edge)

            prev, curr = start, nbr

            while curr not in boundary_nodes:

                nxts = [x for x in IG.neighbors(curr) if x != prev]

                if len(nxts) != 1:
                    raise RuntimeError(
                        f"Unexpected branching at non-boundary node {curr}"
                    )

                nxt = nxts[0]

                visited_edges.add(frozenset((curr, nxt)))
                path.append(nxt)

                prev, curr = curr, nxt

            # curr is the boundary reached
            chains[(start, curr)] = path

    return chains, boundary_nodes


def extract_chains(IG: nx.Graph, nodesDf: pd.DataFrame) -> dict:
    """
    Returns
        {(boundary_low_pk, boundary_high_pk): [nodes ordered by increasing pk_rel]}

    A boundary is a node with degree >= 3 (junction/switch) or degree == 1
    (line end). Each chain is represented canonically:
      - boundary key ordered by increasing pk_rel
      - node list ordered by increasing pk_rel
    """
    boundary_nodes = {n for n, d in IG.degree() if d >= 3 or d == 1}

    chains = {}
    visited_edges = set()

    def pk(node):
        return nodesDf.loc[node, "pk_rel"]

    for start in boundary_nodes:
        for nbr in IG.neighbors(start):
            edge = frozenset((start, nbr))
            if edge in visited_edges:
                continue

            visited_edges.add(edge)

            path = [start, nbr]
            prev, curr = start, nbr

            while curr not in boundary_nodes:
                nxts = [x for x in IG.neighbors(curr) if x != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                visited_edges.add(frozenset((curr, nxt)))
                path.append(nxt)
                prev, curr = curr, nxt

            # Canonical orientation: increasing pk_rel
            if pk(path[0]) > pk(path[-1]):
                path.reverse()

            chains[(path[0], path[-1])] = path

    return chains, boundary_nodes


def chains_key_lookup(chains: dict, chain_key: tuple) -> tuple | None:
    """Your headway dict's chain key might be (a, b) OR (b, a); resolve it."""
    if chain_key in chains:
        return chain_key
    reversed_key = (chain_key[1], chain_key[0])
    if reversed_key in chains:
        return reversed_key
    return None


# --------------------------------------------------------------------------
# 3. Direction inference
# --------------------------------------------------------------------------

def train_direction(route: list, nodesDf: pd.DataFrame) -> str:
    """
    'up' or 'down' from net pk_rel change along the route.
    Uses first and last node of the route that are present in nodesDf.
    """
    pks = [(node, nodesDf.loc[node, "pk_rel"]) for node in route
           if node in nodesDf.index]
    if len(pks) < 2:
        raise ValueError(f"Route too short / not matched to nodesDf: {route}")
    increasing = pks[-1][1] > pks[0][1]
    if UP_IS_INCREASING_PK:
        return "up" if increasing else "down"
    return "down" if increasing else "up"


# --------------------------------------------------------------------------
# 4. Speed-category heuristic -- VERIFY before trusting
# --------------------------------------------------------------------------

def train_speed_category(train_id, route: list, trip_data: dict,
                          nodesDf: pd.DataFrame) -> str:
    """
    Heuristic: if the train's route passes through any node flagged
    stop_slow=1 & stop_fast=0 (a "slow-only" stop) WITHOUT stopping there
    (i.e. it's not in trip_data[train_id]'s station list), classify as
    'fast'. Otherwise 'slow'.
    Spot-check this against trains you know the category of.
    """
    stop_codes = {s[0] for s in trip_data[train_id]}
    for node in route:
        if node not in nodesDf.index:
            continue
        row = nodesDf.loc[node]
        if row.get("stop_slow") == 1 and row.get("stop_fast") == 0:
            if node not in stop_codes:
                return "fast"
    return "slow"


def train_category_label(train_id, route, trip_data, nodesDf) -> str:
    """e.g. 'fast up' -- matches the category strings in your headway dict."""
    return f"{train_speed_category(train_id, route, trip_data, nodesDf)} " \
           f"{train_direction(route, nodesDf)}"


# --------------------------------------------------------------------------
# 5. Map a chain boundary to the nearest real (timetabled) EAN event
# --------------------------------------------------------------------------

def _route_timetable_stop_positions(train_id, route: list, trip_data: dict) -> list[tuple[int, int]]:
    """Return the route indices and timetable-sequence indices of the stops that
    the train actually visits, in the order they appear along the route.

    This is robust to repeated station names and to routes that mix real stops
    with intermediate infrastructure nodes.
    """
    stop_codes = [s[0] for s in trip_data[train_id]]
    positions = []
    stop_cursor = 0

    for route_idx, node in enumerate(route):
        if stop_cursor >= len(stop_codes):
            break
        if node == stop_codes[stop_cursor]:
            positions.append((route_idx, stop_cursor))
            stop_cursor += 1

    return positions


def chain_boundary_event(train_id, route: list, trip_data: dict,
                          boundary_node: str, side: str):
    """
    Map a chain boundary node to the nearest real timetable event along a
    train's route.

    The mapping uses the train's route order and the timetable stop sequence,
    so it remains valid for repeated station names and branched/looped
    networks. The returned event is the departure event of the last stop
    before/at the boundary for side='entry', or the arrival event of the
    first stop at/after the boundary for side='exit'.

    Returns (seq, event_type) or None if not found (e.g. boundary is beyond
    the train's origin/terminus, meaning this train does not actually traverse
    the chain on this side and the caller should skip it).
    """
    if boundary_node not in route:
        return None

    boundary_idx = route.index(boundary_node)
    stop_positions = _route_timetable_stop_positions(train_id, route, trip_data)

    if side == "entry":
        candidates = [pos for pos in stop_positions if pos[0] <= boundary_idx]
        if not candidates:
            return None
        seq = candidates[-1][1]
        if seq == len(trip_data[train_id]) - 1:
            return None
        return (seq, "dep")

    # side == "exit"
    candidates = [pos for pos in stop_positions if pos[0] >= boundary_idx]
    if not candidates:
        return None
    seq = candidates[0][1]
    if seq == 0:
        return None
    return (seq, "arr")


# --------------------------------------------------------------------------
# 6. Assemble headway constraints for every chain
# --------------------------------------------------------------------------


def assemble_headway_constraints(
    trip_data: dict,
    trip_data_enriched: dict,
    routes: dict,
    nodesDf: pd.DataFrame,
    chains: dict,
    headway_dict: dict,
) -> tuple[list[dict], list]:

    def time_to_sec(t):
        return t.hour * 3600 + t.minute * 60 + t.second

    def event_time_sec(train_id, seq, event):
        t = trip_data_enriched[train_id][seq][2 if event == "dep" else 1]
        return time_to_sec(t)

    def category(speed, direction):
        return f"{speed} {direction}"

    chain_keys_in_headways = {k[0] for k in headway_dict}

    constraints = []
    skipped = []

    for chain_key in chain_keys_in_headways:

        resolved = chains_key_lookup(chains, chain_key)
        if resolved is None:
            skipped.append(("unresolved chain", chain_key))
            continue

        boundary_a, boundary_b = resolved

        occupants = []

        # --------------------------------------------------
        # Find all trains using this chain
        # --------------------------------------------------
        for train_id, route in routes.items():

            if boundary_a not in route or boundary_b not in route:
                continue

            idx_a = route.index(boundary_a)
            idx_b = route.index(boundary_b)

            if idx_a < idx_b:
                entry_node, exit_node = boundary_a, boundary_b
            else:
                entry_node, exit_node = boundary_b, boundary_a

            entry_evt = chain_boundary_event(
                train_id,
                route,
                trip_data_enriched,
                entry_node,
                "entry",
            )

            exit_evt = chain_boundary_event(
                train_id,
                route,
                trip_data_enriched,
                exit_node,
                "exit",
            )

            if entry_evt is None or exit_evt is None:
                skipped.append(
                    ("no EAN event on chain", chain_key, train_id)
                )
                continue

            entry_seq, entry_event = entry_evt
            exit_seq, exit_event = exit_evt

            speed = train_speed_category(
                train_id,
                route,
                trip_data,
                nodesDf,
            )

            direction = train_direction(route, nodesDf)

            occupants.append(
                {
                    "train": train_id,
                    "speed": speed,
                    "direction": direction,

                    "entry_seq": entry_seq,
                    "entry_event": entry_event,
                    "entry_time": event_time_sec(
                        train_id,
                        entry_seq,
                        entry_event,
                    ),

                    "exit_seq": exit_seq,
                    "exit_event": exit_event,
                    "exit_time": event_time_sec(
                        train_id,
                        exit_seq,
                        exit_event,
                    ),
                }
            )

        up = [
            o for o in occupants
            if o["direction"] == "up"
        ]

        down = [
            o for o in occupants
            if o["direction"] == "down"
        ]
        #print (f"Chain {chain_key}: {len(up)} up trains, {len(down)} down trains")

        # ==================================================
        # Same-direction constraints
        # ==================================================


        for group in (up, down):

            group.sort(key=lambda x: x["entry_time"])
            for first, second in zip(group, group[1:]):

                cat1 = category(
                    first["speed"],
                    first["direction"],
                )
                cat2 = category(
                    second["speed"],
                    second["direction"],
                )

                key = (chain_key, cat1, cat2)

                if key not in headway_dict:
                    skipped.append(
                        (
                            "category pair missing",
                            chain_key,
                            cat1,
                            cat2,
                        )
                    )
                    continue

                constraints.append(
                    {
                        "train_i": first["train"],
                        "seq_i": first["entry_seq"],
                        "event_i": first["entry_event"],

                        "train_j": second["train"],
                        "seq_j": second["entry_seq"],
                        "event_j": second["entry_event"],

                        "min_headway": headway_dict[key],
                        "resource": chain_key,
                    }
                )

        # ==================================================
        # Opposite-direction constraints
        # ==================================================


        for a in up:
            for b in down:

                # A leaves before B enters
                if a["exit_time"] <= b["entry_time"]:

                    first = (
                        a["train"],
                        a["exit_seq"],
                        a["exit_event"],
                        a["speed"],
                        a["direction"],
                    )

                    second = (
                        b["train"],
                        b["entry_seq"],
                        b["entry_event"],
                        b["speed"],
                        b["direction"],
                    )

                # B leaves before A enters
                elif b["exit_time"] <= a["entry_time"]:

                    first = (
                        b["train"],
                        b["exit_seq"],
                        b["exit_event"],
                        b["speed"],
                        b["direction"],
                    )

                    second = (
                        a["train"],
                        a["entry_seq"],
                        a["entry_event"],
                        a["speed"],
                        a["direction"],
                    )

                else:
                    # Overlapping occupancy without a valid ordering
                    skipped.append(
                        (
                            "overlapping opposite trains",
                            chain_key,
                            a["train"],
                            b["train"],
                        )
                    )
                    continue

                cat1 = category(first[3], first[4])
                cat2 = category(second[3], second[4])

                key = (chain_key, cat1, cat2)

                if key not in headway_dict:
                    skipped.append(
                        (
                            "category pair missing",
                            chain_key,
                            cat1,
                            cat2,
                        )
                    )
                    continue

                constraints.append(
                    {
                        "train_i": first[0],
                        "seq_i": first[1],
                        "event_i": first[2],

                        "train_j": second[0],
                        "seq_j": second[1],
                        "event_j": second[2],

                        "min_headway": headway_dict[key],
                        "resource": chain_key,
                    }
                )

    return constraints, skipped

def generate_perturbation_scenarios(
    G,
    n_scenarios=5,
    entry_delay_mean=300,
    entry_delay_std=60,
    running_delay_mean=0,
    running_delay_std=60,
    perturbation_probability=0.3,
    seed=None,
):
    """
    Generate random node and edge perturbation scenarios.

    Parameters
    ----------
    G : nx.DiGraph
        Scheduled event-activity network.

    n_scenarios : int
        Number of perturbation scenarios to generate.

    entry_delay_mean : float
        Mean entry delay (s).

    entry_delay_std : float
        Standard deviation of entry delay (s).

    running_delay_mean : float
        Mean running-edge perturbation (s).

    running_delay_std : float
        Standard deviation of running-edge perturbation (s).

    seed : int or None
        Random seed.

    Returns
    -------
    node_perturbations : list[dict]
        One dictionary per scenario:
        {entry_node: delay}

    edge_perturbations : list[dict]
        One dictionary per scenario:
        {(u, v): delay}
    """

    rng = np.random.default_rng(seed)

    node_perturbations = []
    edge_perturbations = []

    # --------------------------------------------------
    # Identify first departure node of each train
    # --------------------------------------------------

    entry_nodes = []

    trains = {
        data["train"]
        for _, data in G.nodes(data=True)
        if "train" in data
    }

    for train in trains:
        first_dep = next(
            (
                n
                for n, data in G.nodes(data=True)
                if data.get("train") == train
                and data.get("event") == "dep"
                and data.get("is_stop")
                and data.get("seq") == 0
            ),
            None,
        )

        if first_dep is not None:
            entry_nodes.append(first_dep)

    # --------------------------------------------------
    # Running edges
    # --------------------------------------------------

    running_edges = [
        (u, v)
        for u, v, data in G.edges(data=True)
        if data["kind"] == "running"
    ]

    # --------------------------------------------------
    # Generate scenarios
    # --------------------------------------------------
    reference_duration =  np.mean([G.edges[e]["scheduled_duration"] for e in running_edges])

    edge_delay = {}

    for _ in range(n_scenarios):

        node_delay = {
            node: max(0, rng.normal(entry_delay_mean, entry_delay_std))
            for node in entry_nodes
        }

        for edge in running_edges:
            if rng.random() < perturbation_probability:
                scale = np.sqrt(G.edges[edge]["scheduled_duration"] / reference_duration)
                edge_delay[edge] = rng.normal(
                    running_delay_mean * scale,
                    running_delay_std * scale,
                )
            else:
                edge_delay[edge] = 0.0

        node_perturbations.append(node_delay)
        edge_perturbations.append(edge_delay)

    return node_perturbations, edge_perturbations