from collections import defaultdict

def get_routing_node(node, nodesDf, direction):
    """
    For trains towards Malles, use station side nodes when available.
    For trains towards Merano, always use main station nodes.
    """

    if direction == "malles":

        side_node = f"{node}_side"

        if side_node in nodesDf.index:
            return side_node

    return node

# --------------------------------------------------
# Route builder
# --------------------------------------------------

def build_route(nodesDf, edgesDf, start, end):

    successors = defaultdict(list)

    for _, edge in edgesDf.iterrows():

        successors[edge["node_from"]].append(edge["node_to"])
        successors[edge["node_to"]].append(edge["node_from"])

        direction = (
        "malles"
        if nodesDf.loc[start, "pk_rel"] < nodesDf.loc[end, "pk_rel"]
        else "merano"
    )

    route_start = get_routing_node(start, nodesDf, direction)
    route_end = get_routing_node(end, nodesDf, direction)

    route = [route_start]

    current = route_start
    visited = {route_start}

    while current != route_end:

        pk_current = nodesDf.loc[current, "pk_rel"]

        # only move towards destination
        candidates = [
            n for n in successors[current]
            if (
                nodesDf.loc[n, "pk_rel"] > pk_current
                if direction == "malles"
                else nodesDf.loc[n, "pk_rel"] < pk_current
            )
        ]

        if not candidates:
            raise ValueError(
                f"No valid successor from {current} towards {route_end}"
            )

        # at branching points choose preferred track
        if len(candidates) > 1:

            preferred = [
                n for n in candidates
                if (
                    nodesDf.loc[n, "y"] != 0
                    if direction == "malles"
                    else nodesDf.loc[n, "y"] == 0
                )
            ]

            if preferred:
                candidates = preferred

        if len(candidates) > 1:
            print(current)
            print("Candidates:", candidates)
        next_node = candidates[0]

        if next_node in visited:
            raise ValueError(
                f"Loop detected: {current} -> {next_node}"
            )

        route.append(next_node)
        visited.add(next_node)

        current = next_node

    return route


# --------------------------------------------------
# Build routing dictionary
# --------------------------------------------------



def get_signal_nodes_on_route(nodesDf, edgesDf, route_nodes, is_forward):

    valid_dirs = ["malles", "both"] if is_forward else ["merano", "both"]

    signal_nodes = []

    for n in route_nodes:

        node = nodesDf.loc[n]

        if node["valid_dir"] not in valid_dirs:
            continue
        
        signal_nodes.append(n)

    return signal_nodes