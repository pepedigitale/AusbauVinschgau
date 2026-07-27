"""
=============================================================================
VINSCHGERBAHN – Block-Section Model + Interactive Bokeh Visualization
=============================================================================
Builds a custom dictionary-based infrastructure model from nodes.csv and
edges.csv, derives directed block sections for both directions, and produces
an interactive HTML visualization with:
  - Direction toggle  (Merano→Malles  /  Malles→Merano)
  - Max-speed overlay toggle
  - Color-coded block sections
  - Hover tooltips on nodes and block segments

Usage:
    python vinschgau_infra.py
    python vinschgau_infra.py --nodes nodes.csv --edges edges.csv --output out.html

CSV requirements (semicolon-separated, comma decimal):
    nodes.csv : node_id, name, valid_dir, pk_rel, pk_abs, node_type,
                stop_slow, stop_fast, y
    edges.csv : edge_id, node_from, node_to,
                max_speed_dir_malles, max_speed_dir_merano
=============================================================================
"""

import argparse
import os
import math
from collections import defaultdict

import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.models import (
    ColumnDataSource, HoverTool, Toggle, CustomJS,
    Div, RadioButtonGroup, Spacer
)
from bokeh.layouts import column, row
from bokeh.palettes import Category20
from bokeh.transform import factor_cmap
from routing import (build_route,get_signal_nodes_on_route)

# =============================================================================
# 1. CSV LOADING
# =============================================================================

def load_csv(path: str, active_scenarios=None) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="cp1252", decimal=",")

    if active_scenarios is not None:
        df = df[
            df["scenario"].fillna("").apply(
                lambda s: bool(set(map(str.strip, s.split(","))) & active_scenarios)
            )
        ]

    return df


def validate(nodes_df, edges_df):
    req_nodes = {"node_id", "name", "valid_dir", "pk_rel", "node_type", "y"}
    req_edges = {"edge_id", "node_from", "node_to",
                 "max_speed_dir_malles", "max_speed_dir_merano"}
    missing_n = req_nodes - set(nodes_df.columns)
    missing_e = req_edges - set(edges_df.columns)
    if missing_n:
        raise ValueError(f"nodes.csv missing columns: {missing_n}")
    if missing_e:
        raise ValueError(f"edges.csv missing columns: {missing_e}")
    node_ids = set(nodes_df["node_id"])
    for col in ("node_from", "node_to"):
        unknown = set(edges_df[col]) - node_ids
        if unknown:
            print(f"  WARNING: edges.csv references unknown node_id(s) in "
                  f"'{col}': {unknown}")


def get_unique_scenarios(nodes_df: pd.DataFrame) -> set:
    """Extract all unique scenario tags from the semicolon-separated scenario field."""
    if "scenario" not in nodes_df.columns:
        return set()

    scenarios = set()
    for raw_value in nodes_df["scenario"].fillna("").astype(str):
        for token in raw_value.split(","):
            token = token.strip()
            if token:
                scenarios.add(token)
    return scenarios


# =============================================================================
# 2. INFRASTRUCTURE DICTIONARY
# =============================================================================

def build_infrastructure(nodes_df: pd.DataFrame,
                         edges_df: pd.DataFrame) -> dict:
    """
    Returns:
        infrastructure = {
            "nodes": { node_id: { all CSV attributes } },
            "edges": { edge_id: { all CSV attributes } },
            "adj"  : { node_id: [ (neighbour_id, edge_id), ... ] }
        }
    """
    nodes = {}
    for _, r in nodes_df.iterrows():
        d = r.to_dict()
        if pd.isna(d.get("valid_dir")):
            d["valid_dir"] = None
        node_type = d.get("node_type")
        if pd.isna(node_type):
            d["node_type"] = ""
        else:
            d["node_type"] = str(node_type).strip()
        nodes[d["node_id"]] = d

    edges = {}
    adj = defaultdict(list)
    for _, r in edges_df.iterrows():
        d = r.to_dict()
        eid = d["edge_id"]
        edges[eid] = d
        nf, nt = d["node_from"], d["node_to"]
        # Only add to adjacency if both endpoints exist in nodes
        if nf in nodes and nt in nodes:
            adj[nf].append((nt, eid))
            adj[nt].append((nf, eid))

    return {"nodes": nodes, "edges": edges, "adj": dict(adj)}


# =============================================================================
# 3. BLOCK SECTION EXTRACTION
# =============================================================================

DIRECTIONS = {
    "malles": ("ME", "MAL", True),
    "merano": ("MAL", "ME", False),
}


def extract_blocks(infra):
    """
    Build signalling blocks directly from the canonical railway routes.

    Each block is the infrastructure between two consecutive signalling
    nodes along the operational route.

    Returns
    -------
    blocks : dict
    block_connections : dict
    """

    nodes = infra["nodes"]
    edges = infra["edges"]

    # edge lookup
    edge_lookup = {}

    for eid, e in edges.items():

        edge_lookup[(e["node_from"], e["node_to"])] = eid
        edge_lookup[(e["node_to"], e["node_from"])] = eid

    blocks = {}
    block_connections = {}

    for direction, (origin, destination, is_forward) in DIRECTIONS.items():

        route = build_route(origin, destination)

        signal_nodes = get_signal_nodes_on_route(route, is_forward)

        previous_block = None

        counter = 1

        for i in range(len(signal_nodes) - 1):

            entry = signal_nodes[i]
            exit = signal_nodes[i + 1]

            route_index = {
                node: i
                for i, node in enumerate(route)
            }

            i0 = route_index[entry]
            i1 = route_index[exit]

            block_nodes = route[i0:i1 + 1]

            block_edges = []

            length = 0.0

            edge_speeds_malles = []
            edge_speeds_merano = []

            for n0, n1 in zip(block_nodes[:-1], block_nodes[1:]):

                eid = edge_lookup[(n0, n1)]

                block_edges.append(eid)

                e = edges[eid]

                length += abs(
                    nodes[n1]["pk_rel"]
                    - nodes[n0]["pk_rel"]
                )

                sm = e.get("max_speed_dir_malles")
                sr = e.get("max_speed_dir_merano")

                edge_speeds_malles.append(
                    float(sm) if pd.notna(sm) else None
                )

                edge_speeds_merano.append(
                    float(sr) if pd.notna(sr) else None
                )

            bid = f"{direction}_BLK_{counter:03d}"

            blocks[bid] = {

                "direction": direction,

                "entry_signal": entry,

                "exit_signal": exit,

                "nodes": block_nodes,

                "edges": block_edges,

                "length_km": round(length, 4),

                "edge_speeds_malles": edge_speeds_malles,

                "edge_speeds_merano": edge_speeds_merano,

            }

            if previous_block is not None:
                block_connections.setdefault(previous_block, []).append(bid)

            block_connections.setdefault(bid, [])

            previous_block = bid

            counter += 1

    return blocks, block_connections


# =============================================================================
# 4. SUMMARY PRINTING
# =============================================================================

def print_summary(infra, blocks, connections):
    print("\n=== Infrastructure ===")
    print(f"  Nodes : {len(infra['nodes'])}")
    print(f"  Edges : {len(infra['edges'])}")
    type_counts = defaultdict(int)
    for n in infra["nodes"].values():
        type_counts[n.get("node_type", "unknown")] += 1
    for t, c in sorted(type_counts.items(), key=lambda x: str(x[0])):
        print(f"    {str(t):20s}: {c}")

    print("\n=== Block Sections ===")
    for d in DIRECTIONS:
        d_blocks = {k: v for k, v in blocks.items() if v["direction"] == d}
        total_km = sum(b["length_km"] for b in d_blocks.values())
        print(f"  {d:8s}: {len(d_blocks):3d} blocks  |  "
              f"total path length {total_km:.3f} km")
    print()


# =============================================================================
# 5. BOKEH VISUALIZATION
# =============================================================================

# Palette: 20 distinct colors, cycling if more blocks than colors
PALETTE = Category20[20]


def _block_color(idx: int) -> str:
    return PALETTE[idx % len(PALETTE)]


def _speed_color(speed) -> str:
    """Map speed to a color: green (fast) → orange → red (slow)."""
    if speed is None:
        return "#cccccc"
    speed = float(speed)
    if speed >= 120:
        return "#16a34a"
    elif speed >= 100:
        return "#65a30d"
    elif speed >= 70:
        return "#ca8a04"
    elif speed >= 60:
        return "#ea580c"
    else:
        return "#dc2626"


def build_visualization(infra, blocks, connections, output_path):
    nodes = infra["nodes"]
    edges = infra["edges"]

    # ------------------------------------------------------------------
    # Pre-compute per-direction block segment data for rendering
    # ------------------------------------------------------------------
    # Each block is drawn as a polyline through its node coordinates.
    # We store one entry per block: xs (list), ys (list), plus metadata.

    dir_data = {}
    for direction in DIRECTIONS:
        spd_edge_key = f"edge_speeds_{direction}"

        # --- Block-color layer: one polyline per block (unchanged) ---
        segs_xs    = []
        segs_ys    = []
        segs_color = []
        segs_bid   = []
        segs_entry = []
        segs_exit  = []
        segs_length = []
        segs_gradient = []

        # --- Speed-color layer: one polyline per EDGE so each edge gets
        #     its own colour independently of its block neighbours ---
        spd_xs      = []
        spd_ys      = []
        spd_color   = []
        spd_bid     = []
        spd_speed   = []
        spd_length  = []
        spd_gradient = []

        d_blocks = {k: v for k, v in blocks.items()
                    if v["direction"] == direction}

        for idx, (bid, bdata) in enumerate(d_blocks.items()):
            path_nodes = bdata["nodes"]
            edge_speeds = bdata.get(spd_edge_key, [])
            for i, eid in enumerate(bdata["edges"]):
                n0 = path_nodes[i]
                n1 = path_nodes[i + 1]
                if n0 not in nodes or n1 not in nodes:
                    continue
                e = edges.get(eid, {})
                edge_length = e.get("length")
                edge_gradient = e.get("gradient")
                segs_xs.append([nodes[n0]["pk_rel"], nodes[n1]["pk_rel"]])
                segs_ys.append([nodes[n0]["y"],      nodes[n1]["y"]])
                segs_color.append(_block_color(idx))
                segs_bid.append(bid)
                segs_entry.append(bdata["entry_signal"])
                segs_exit.append(bdata["exit_signal"])
                segs_length.append(
                    str(edge_length) + " km" if pd.notna(edge_length) else "n/a"
                )
                segs_gradient.append(
                    str(edge_gradient) if pd.notna(edge_gradient) else "n/a"
                )

                spd = edge_speeds[i] if i < len(edge_speeds) else None
                spd_xs.append([nodes[n0]["pk_rel"], nodes[n1]["pk_rel"]])
                spd_ys.append([nodes[n0]["y"],      nodes[n1]["y"]])
                spd_color.append(_speed_color(spd))
                spd_bid.append(bid)
                spd_speed.append(str(int(spd)) if spd is not None else "n/a")
                spd_length.append(str(edge_length) if pd.notna(edge_length) else "n/a")
                spd_gradient.append(str(edge_gradient) if pd.notna(edge_gradient) else "n/a")

        dir_data[direction] = dict(
            # block-color layer
            xs=segs_xs, ys=segs_ys,
            color=segs_color,
            bid=segs_bid, entry=segs_entry, exit=segs_exit,
            length=segs_length, gradient=segs_gradient,
            # speed-color layer (per-edge)
            spd_xs=spd_xs, spd_ys=spd_ys,
            spd_color=spd_color, spd_bid=spd_bid, spd_speed=spd_speed,
            spd_length=spd_length, spd_gradient=spd_gradient,
        )

    # ------------------------------------------------------------------
    # Node data (shared across directions)
    # ------------------------------------------------------------------
    node_x, node_y, node_id_list = [], [], []
    node_name, node_type, node_vdir = [], [], []
    node_altitude = []
    node_size, node_color, node_shape = [], [], []

    NODE_VIZ = {
        "LdS":         {"size": 16, "color": "#1d4ed8", "shape": "square"},
        "SC":          {"size": 10, "color": "#7c3aed", "shape": "diamond"},
        "signal_main": {"size":  7, "color": "#16a34a", "shape": "triangle"},
        "signal_side": {"size":  7, "color": "#f59e0b", "shape": "triangle"},
        "PL":          {"size":  8, "color": "#dc2626", "shape": "diamond"},
        "Tunnel":      {"size":  8, "color": "#6b7280", "shape": "inverted_triangle"},
        "Sbl":         {"size":  7, "color": "#d97706", "shape": "circle"},
        "km":          {"size":  5, "color": "#d1d5db", "shape": "circle"},
    }
    default_viz = {"size": 6, "color": "#9ca3af", "shape": "circle"}

    for nid, nd in nodes.items():
        node_x.append(nd["pk_rel"])
        node_y.append(nd["y"])
        node_id_list.append(nid)
        node_name.append(nd.get("name", nid))
        node_type_value = str(nd.get("node_type", "")).strip()
        node_type.append(node_type_value)
        node_vdir.append(nd.get("valid_dir") or "—")
        node_altitude.append(nd.get("altitude", "n/a"))
        viz = NODE_VIZ.get(node_type_value, default_viz)
        node_size.append(viz["size"])
        node_color.append(viz["color"])
        node_shape.append(viz["shape"])

    node_source = ColumnDataSource(dict(
        x=node_x, y=node_y,
        node_id=node_id_list, name=node_name,
        node_type=node_type, valid_dir=node_vdir,
        altitude=node_altitude,
        size=node_size, color=node_color,
    ))

    # ------------------------------------------------------------------
    # Bokeh figure
    # ------------------------------------------------------------------
    p = figure(
        width=1600, height=500,
        title="Vinschgerbahn – Block Sections",
        x_axis_label="Chainage pk_rel [km]",
        y_axis_label="Track level",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )
    p.title.text_font_size = "14pt"
    p.yaxis.ticker = [-1, 0, 1]
    p.yaxis.major_label_overrides = {-1: "siding (−1)", 0: "mainline", 1: "siding (+1)"}
    p.y_range.start = -1.6
    p.y_range.end   =  1.6
    p.grid.grid_line_alpha = 0.3

    # ------------------------------------------------------------------
    # Block segment glyphs — one MultiLine per direction
    # ------------------------------------------------------------------
    sources = {}
    block_renderers = {}
    speed_renderers = {}

    for direction in DIRECTIONS:
        dd = dir_data[direction]

        # Block-color source (one polyline per block)
        src_blk = ColumnDataSource(dict(
            xs=dd["xs"], ys=dd["ys"],
            color=dd["color"],
            bid=dd["bid"],
            entry=dd["entry"],
            exit=dd["exit"],
            length=dd["length"],
            gradient=dd["gradient"],
        ))
        # Speed-color source (one polyline per EDGE for fine-grained colouring)
        src_spd = ColumnDataSource(dict(
            xs=dd["spd_xs"], ys=dd["spd_ys"],
            color=dd["spd_color"],
            bid=dd["spd_bid"],
            speed=dd["spd_speed"],
            length=dd["spd_length"],
            gradient=dd["spd_gradient"],
        ))
        sources[direction] = (src_blk, src_spd)

        r_blk = p.multi_line(
            xs="xs", ys="ys",
            line_color="color",
            line_width=5,
            line_alpha=0.85,
            source=src_blk,
            visible=(direction == "malles"),  # malles shown by default
        )
        r_spd = p.multi_line(
            xs="xs", ys="ys",
            line_color="color",
            line_width=5,
            line_alpha=0.85,
            source=src_spd,
            visible=False,  # hidden until speed toggle is ON
        )
        block_renderers[direction] = r_blk
        speed_renderers[direction] = r_spd

    # Hover for block segments
    hover_blk = HoverTool(
        renderers=[block_renderers["malles"], block_renderers["merano"]],
        tooltips=[
            ("Block",        "@bid"),
            ("Entry signal", "@entry"),
            ("Exit signal",  "@exit"),
            ("Edge length",  "@length"),
            ("Gradient",     "@gradient"),
        ],
        line_policy="nearest",
    )
    hover_spd = HoverTool(
        renderers=[speed_renderers["malles"], speed_renderers["merano"]],
        tooltips=[
            ("Block",     "@bid"),
            ("Min speed", "@speed km/h"),
            ("Edge length", "@length"),
            ("Gradient",   "@gradient"),
        ],
        line_policy="nearest",
    )
    p.add_tools(hover_blk, hover_spd)

    # ------------------------------------------------------------------
    # Node scatter (drawn last so it sits on top)
    # ------------------------------------------------------------------
    r_nodes = p.scatter(
        x="x", y="y",
        size="size",
        color="color",
        source=node_source,
        marker="circle",
        line_color="white",
        line_width=0.5,
        alpha=0.9,
    )
    p.add_tools(HoverTool(
        renderers=[r_nodes],
        tooltips=[
            ("ID",        "@node_id"),
            ("Name",      "@name"),
            ("Type",      "@node_type"),
            ("valid_dir", "@valid_dir"),
            ("Altitude",  "@altitude"),
            ("pk_rel",    "@x{0.000} km"),
        ],
    ))

    # Station labels
    lds_ids = [i for i, t in enumerate(node_type) if str(t).strip() == "LdS"]
    lbl_x = [node_x[i] for i in lds_ids]
    lbl_y = [node_y[i] + 0.05 for i in lds_ids]
    lbl_names = [node_name[i] for i in lds_ids]
    lbl_src   = ColumnDataSource(dict(x=lbl_x, y=lbl_y, name=lbl_names))
    p.text(x="x", y="y", text="name",
           source=lbl_src,
           text_font_size="8pt", text_align="center",
           text_color="#1e3a5f", text_font_style="bold")

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    # Direction radio button
    dir_radio = RadioButtonGroup(
        labels=["Merano → Malles", "Malles → Merano"],
        active=0,
        width=280,
    )

    # Speed toggle
    spd_toggle = Toggle(
        label="Show max speed (colour)",
        button_type="warning",
        active=False,
        width=220,
    )

    # JS callback for direction + speed interaction
    cb_code = """
        const active_dir   = dir_radio.active === 0 ? 'malles' : 'merano';
        const inactive_dir = dir_radio.active === 0 ? 'merano' : 'malles';
        const show_speed   = spd_toggle.active;

        // block-color renderers
        blk_malles.visible = (!show_speed) && (active_dir === 'malles');
        blk_merano.visible = (!show_speed) && (active_dir === 'merano');

        // speed-color renderers
        spd_malles.visible = show_speed && (active_dir === 'malles');
        spd_merano.visible = show_speed && (active_dir === 'merano');
    """

    callback_args = dict(
        dir_radio=dir_radio,
        spd_toggle=spd_toggle,
        blk_malles=block_renderers["malles"],
        blk_merano=block_renderers["merano"],
        spd_malles=speed_renderers["malles"],
        spd_merano=speed_renderers["merano"],
    )

    dir_radio.js_on_change("active",  CustomJS(args=callback_args, code=cb_code))
    spd_toggle.js_on_change("active", CustomJS(args=callback_args, code=cb_code))

    # Legend div
    legend_html = """
    <div style='font-size:12px; padding:6px 10px; line-height:1.8'>
    <b>Node types</b><br>
    <span style='color:#1d4ed8'>■</span> Station (LdS) &nbsp;
    <span style='color:#7c3aed'>◆</span> Switch (SC) &nbsp;
    <span style='color:#16a34a'>▲</span> Signal main &nbsp;
    <span style='color:#f59e0b'>▲</span> Signal side &nbsp;
    <span style='color:#dc2626'>◆</span> Level crossing (PL) &nbsp;
    <span style='color:#6b7280'>▼</span> Tunnel<br>
    <b>Speed colours (when speed overlay is ON)</b><br>
    <span style='color:#16a34a'>━</span> ≥120 km/h &nbsp;
    <span style='color:#65a30d'>━</span> ≥100 &nbsp;
    <span style='color:#ca8a04'>━</span> ≥70 &nbsp;
    <span style='color:#ea580c'>━</span> ≥60 &nbsp;
    <span style='color:#dc2626'>━</span> &lt;60 km/h
    </div>
    """
    legend_div = Div(text=legend_html, width=900)

    # ------------------------------------------------------------------
    # Layout and export
    # ------------------------------------------------------------------
    controls = row(
        Div(text="<b>Direction:</b>", width=80,
            styles={"line-height": "2.2", "font-size": "13px"}),
        dir_radio,
        Spacer(width=40),
        spd_toggle,
    )
    layout = column(controls, p, legend_div)

    output_file(output_path, title="Vinschgerbahn Infrastructure")
    save(layout)
    print(f"Visualization saved to: {output_path}")


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Vinschgerbahn block-section model and Bokeh visualizer."
    )
    parser.add_argument("--nodes", default="nodes_double.csv")
    parser.add_argument("--edges", default="edges_double.csv")
    parser.add_argument("--scenario", default="1a")
    parser.add_argument("--output", default="vinschgau_infra1a.html")
    args = parser.parse_args()

    SCENARIOS = {"0": {"base", "existing"}, 
             "1a": {"base", "existing", "dt_me_lag", "dt_tel_pla", "dt_natkomp_sta", "dt_sta_cia", "dt_cold_sblLac", "dt_lasa_oris" },
             "1b": {"base", "existing"},
             "2a": {"base", "existing"}
}

    script_dir = os.path.dirname(os.path.abspath(__file__))

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(script_dir, p)

    nodes_path = resolve(args.nodes)
    edges_path = resolve(args.edges)
    output_path = resolve(args.output)

    raw_nodes_df = pd.read_csv(nodes_path, sep=";", encoding="cp1252", decimal=",")
    SCENARIOS["all"] = get_unique_scenarios(raw_nodes_df)

    ACTIVE_SCENARIOS = SCENARIOS[args.scenario]

    print(f"Loading nodes : {nodes_path}")
    print(f"Loading edges : {edges_path}")

    nodes_df = load_csv(nodes_path, ACTIVE_SCENARIOS)
    edges_df = load_csv(edges_path, ACTIVE_SCENARIOS)
    from collections import defaultdict

    successors = defaultdict(list)

    for _, edge in edges_df.iterrows():
        successors[edge["node_from"]].append(edge["node_to"])
        successors[edge["node_to"]].append(edge["node_from"])
    validate(nodes_df, edges_df)

    infra = build_infrastructure(nodes_df, edges_df)
    blocks, connections = extract_blocks(infra)

    print_summary(infra, blocks, connections)

    build_visualization(infra, blocks, connections, output_path)


if __name__ == "__main__":
    main()
