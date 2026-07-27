"""
railml_timetable.py
────────────────────────────────────────────────────────────────────────────
Graphical timetable (Marey / distance-time diagram) from a railML 2.2 file.

USAGE
-----
1. Place this script in the same folder as your railML file.
2. Set RAILML_FILE and LINE_ID below.
3. Set the colour map for categories you care about (CATEGORY_COLORS).
4. Run:  python railml_timetable.py

CONFIGURATION (edit the block below)
--------------------------------------
"""

# ── USER CONFIGURATION ────────────────────────────────────────────────────────

file_name = "RidottoStamm6"

RAILML_FILE  = f"{file_name}.railml"   # ← path to your exported railML file
LINE_ID      = "ln_Vinschgerbahn"      # ← id attribute of the <line> to plot

# Colour coding by category id  (anything not listed → light grey)
CATEGORY_COLORS = {
    "cat_R-SAD": "#E53935",   # red
    "cat_R-TI":  "#2ECC71",   # green
    "cat_IRE":   "#2196F3",   # blue
}
DEFAULT_COLOR = "#D3D3D3"     # light grey for all other categories

# to convert different formats/languages

STATION_ABBREVIATIONS = {
    "Meran": "ME",
    "Lagundo": "LAG",
    "Algund": "LAG",
    "Marlengo": "MAR",
    "Marling": "MAR",
    "Tel": "TEL",
    "Töll": "TEL",
    "Rablà": "RAB",
    "Rabland": "RAB",
    "Plaus": "PLA",
    "Naturno": "NAT",
    "Naturns": "NAT",
    "Stava": "STA",
    "Staben": "STA",
    "Ciardes": "CIA",
    "Tschars": "CIA",
    "Castelbello": "CAB",
    "Kastelbell": "CAB",
    "Laces": "LAC",
    "Latsch": "LAC",
    "Coldrano": "COLD",
    "Goldrain": "COLD",
    "Silandro": "SIL",
    "Schlanders": "SIL",
    "Lasa": "LASA",
    "Laas": "LASA",
    "Eyrs": "ORIS",
    "Oris": "ORIS",
    "Spondinig": "SPON",
    "Spondigna": "SPON",
    "Schluderns": "SLU",
    "Sluderno": "SLU",
    "Mals": "MAL",
    "Malles": "MAL",
}

# Time window to display  (HH:MM on a day when trains run)
TIME_START = "10:00"
TIME_END   = "12:00"

# Figure size in inches
FIG_WIDTH  = 24
FIG_HEIGHT = 14

# ── END USER CONFIGURATION ────────────────────────────────────────────────────

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
import numpy as np
import sys
import os

# railML 2.2 namespace
NS = "http://www.railml.org/schemas/2013"
def tag(local):
    return f"{{{NS}}}{local}"


# ─────────────────────────────────────────────────────────────────────────────
# 1. PARSE THE FILE
# ─────────────────────────────────────────────────────────────────────────────

def load_tree(path):
    if not os.path.exists(path):
        sys.exit(f"ERROR: railML file not found: '{path}'")
    print(f"Parsing {path} …")
    return ET.parse(path).getroot()


# ─────────────────────────────────────────────────────────────────────────────
# 2. COLLECT OCP IDs THAT BELONG TO THE LINE
# ─────────────────────────────────────────────────────────────────────────────

def get_line_track_refs(root, line_id):
    """Return set of track ids referenced by the given line."""
    for line in root.iter(tag("line")):
        if line.get("id") == line_id:
            return {tr.get("ref") for tr in line.findall(tag("trackRef"))}
    sys.exit(f"ERROR: line id '{line_id}' not found in <trackGroups>.")


def get_ocp_positions(root, track_ids):
    """
    For every track in track_ids, collect:
      ocp_id → absPos (metres, as float)
    from trackBegin, trackEnd, and crossSections.

    Returns dict  {ocp_id: absPos_m}
    """
    positions = {}

    for track in root.iter(tag("track")):
        if track.get("id") not in track_ids:
            continue
        topo = track.find(tag("trackTopology"))
        if topo is None:
            continue

        def record(node):
            if node is None:
                return
            mn = node.find(tag("macroscopicNode"))
            ocp_ref = (mn.get("ocpRef") if mn is not None
                       else node.get("ocpRef"))
            abs_pos = node.get("absPos")
            if ocp_ref and abs_pos:
                positions[ocp_ref] = float(abs_pos)

        record(topo.find(tag("trackBegin")))
        record(topo.find(tag("trackEnd")))
        cs_parent = topo.find(tag("crossSections"))
        if cs_parent is not None:
            for cs in cs_parent.findall(tag("crossSection")):
                record(cs)

    return positions


def get_passenger_ocps(root, ocp_positions):
    """
    From ocp_positions keep only those OCPs where propService has
    passenger='true'.  Returns dict {ocp_id: (name, km_position)}.
    """
    result = {}
    for ocp in root.iter(tag("ocp")):
        ocp_id = ocp.get("id")
        if ocp_id not in ocp_positions:
            continue
        ps = ocp.find(tag("propService"))
        if ps is None:
            continue
        if ps.get("passenger", "false").lower() != "true":
            continue
        name = ocp.get("name", ocp_id)
        km   = ocp_positions[ocp_id] / 1000.0
        result[ocp_id] = (name, km)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. PARSE TIMETABLE
# ─────────────────────────────────────────────────────────────────────────────

def parse_time(time_str):
    """
    Convert 'HH:MM:SS' or 'HH:MM:SS.ss' to fractional hours (float).
    Returns None if time_str is None or empty.
    """
    if not time_str:
        return None
    # strip sub-seconds for strptime
    ts = time_str.split(".")[0]
    try:
        t = datetime.strptime(ts, "%H:%M:%S")
    except ValueError:
        return None
    return t.hour + t.minute / 60.0 + t.second / 3600.0


def parse_time_datetime(time_str):
    """
    Convert 'HH:MM:SS' or 'HH:MM:SS.ss' to datetime object with 1900-01-01 base.
    Returns None if time_str is None or empty.
    """
    if not time_str:
        return None
    ts = time_str.split(".")[0]
    try:
        t = datetime.strptime(ts, "%H:%M:%S")
        return datetime(1900, 1, 1, t.hour, t.minute, t.second)
    except ValueError:
        return None


def get_train_paths(root, line_id, passenger_ocps):
    """
    Returns a list of train path dicts:
    {
        'train_id':    str,
        'category_id': str or None,
        'stops': [
            {'ocp_id': str, 'name': str, 'km': float,
             'arr': float|None, 'dep': float|None},
            ...
        ]
    }
    Only trainParts that have at least one sectionTT with lineRef==line_id
    are included.  Only stops at passenger OCPs on the line are kept.
    """
    # Build category map from trainPart id → category id
    # (categoryRef lives on <trainPart> directly)

    paths = []

    for tp in root.iter(tag("trainPart")):
        # ── check if this trainPart runs on our line ──────────────────────
        on_line = any(
            stt.get("lineRef") == line_id
            for stt in tp.iter(tag("sectionTT"))
        )
        if not on_line:
            continue

        tp_id       = tp.get("id", "?")
        category_id = tp.get("categoryRef")

        # ── collect stops ─────────────────────────────────────────────────
        stops = []
        for ocptt in tp.iter(tag("ocpTT")):
            ocp_ref = ocptt.get("ocpRef")
            if ocp_ref not in passenger_ocps:
                continue                          # skip non-passenger / off-line

            name, km = passenger_ocps[ocp_ref]

            times_el = ocptt.find(tag("times"))
            arr_str  = times_el.get("arrival")   if times_el is not None else None
            dep_str  = times_el.get("departure")  if times_el is not None else None

            arr = parse_time(arr_str)
            dep = parse_time(dep_str)

            # graceful fallback: first stop may have no arrival,
            # last stop may have no departure
            if arr is None and dep is not None:
                arr = dep
            if dep is None and arr is not None:
                dep = arr

            if arr is None and dep is None:
                continue

            stops.append({
                "ocp_id": ocp_ref,
                "name":   name,
                "km":     km,
                "arr":    arr,
                "dep":    dep,
            })

        if len(stops) < 2:
            continue

        paths.append({
            "train_id":    tp_id,
            "category_id": category_id,
            "stops":       stops,
        })

    return paths


def get_trip_data(root, line_id, passenger_ocps):
    """
    Extract trip data in the format:
    {trip_id: [(station_name, arrival_datetime, departure_datetime), ...], ...}

    trip_id is an integer counter starting from 1.
    Only trainParts that have at least one sectionTT with lineRef==line_id
    are included.  Only stops at passenger OCPs on the line are kept.
    """
    trip_data = {}
    trip_id = 1

    for tp in root.iter(tag("trainPart")):
        # check if this trainPart runs on our line
        on_line = any(
            stt.get("lineRef") == line_id
            for stt in tp.iter(tag("sectionTT"))
        )
        if not on_line:
            continue

        trip_stops = []
        for ocptt in tp.iter(tag("ocpTT")):
            ocp_ref = ocptt.get("ocpRef")
            if ocp_ref not in passenger_ocps:
                continue

            name, km = passenger_ocps[ocp_ref]

            times_el = ocptt.find(tag("times"))
            arr_str  = times_el.get("arrival")   if times_el is not None else None
            dep_str  = times_el.get("departure")  if times_el is not None else None

            arr = parse_time_datetime(arr_str)
            dep = parse_time_datetime(dep_str)

            # graceful fallback: first stop may have no arrival,
            # last stop may have no departure
            if arr is None and dep is not None:
                arr = dep
            if dep is None and arr is not None:
                dep = arr

            if arr is None and dep is None:
                continue

            name = STATION_ABBREVIATIONS.get(name, name)
            trip_stops.append((name, arr, dep))

        if trip_stops:
            trip_data[trip_id] = trip_stops
            trip_id += 1

    return trip_data


# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOT
# ─────────────────────────────────────────────────────────────────────────────

def time_str_to_hours(s):
    h, m = s.split(":")
    return int(h) + int(m) / 60.0


def plot_timetable(passenger_ocps, train_paths):
    # ── build sorted station list (by km) ────────────────────────────────
    # keep only stations that appear in at least one train path
    used_ocp_ids = {s["ocp_id"] for p in train_paths for s in p["stops"]}
    stations = sorted(
        [(ocp_id, name, km)
         for ocp_id, (name, km) in passenger_ocps.items()
         if ocp_id in used_ocp_ids],
        key=lambda x: x[2]
    )

    if not stations:
        sys.exit("ERROR: No passenger stops found on this line after filtering.")

    km_map = {ocp_id: km for ocp_id, name, km in stations}

    # ── time window ───────────────────────────────────────────────────────
    t_start = time_str_to_hours(TIME_START)
    t_end   = time_str_to_hours(TIME_END)

    # ── figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # ── draw train paths ──────────────────────────────────────────────────
    plotted_categories = set()

    for path in train_paths:
        color = CATEGORY_COLORS.get(path["category_id"], DEFAULT_COLOR)
        stops = path["stops"]

        # Build x (time) and y (km) arrays with dwell segments
        x_vals, y_vals = [], []
        for stop in stops:
            arr = stop["arr"]
            dep = stop["dep"]
            km  = stop["km"]
            # arrival point
            x_vals.append(arr)
            y_vals.append(km)
            # departure point (creates flat horizontal segment for dwell time)
            x_vals.append(dep)
            y_vals.append(km)

        # clip to time window: only draw if any point falls within window
        in_window = any(t_start <= x <= t_end for x in x_vals)
        if not in_window:
            continue

        ax.plot(x_vals, y_vals,
                color=color,
                linewidth=1.5,
                alpha=0.85,
                solid_capstyle="round",
                zorder=2)

        plotted_categories.add(path["category_id"])

    # ── y-axis: stations ──────────────────────────────────────────────────
    y_ticks  = [km  for _, _, km  in stations]
    y_labels = [name for _, name, _ in stations]

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.invert_yaxis()   # origin (smallest km = start) at top

    # ── horizontal grid lines at each station ─────────────────────────────
    for km in y_ticks:
        ax.axhline(km, color="#CCCCCC", linewidth=0.5, zorder=1)

    # ── x-axis: time ──────────────────────────────────────────────────────
    ax.set_xlim(t_start, t_end)

    def fmt_time(x, pos):
        h = int(x) % 24
        m = int(round((x - int(x)) * 60)) % 60
        return f"{h:02d}:{m:02d}"

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1/4))   # every 15 min
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1/12))  # every 5 min
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_time))
    ax.tick_params(axis="x", which="major", labelsize=10, rotation=45)
    ax.tick_params(axis="x", which="minor", length=3)

    # ── vertical grid lines ───────────────────────────────────────────────
    ax.xaxis.grid(True, which="major", color="#DDDDDD", linewidth=0.7, zorder=1)
    ax.xaxis.grid(True, which="minor", color="#EEEEEE", linewidth=0.3, zorder=1)

    # ── labels ────────────────────────────────────────────────────────────
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Distance (km)", fontsize=11)
    ax.set_title(
        f"Graphical Timetable — Line {LINE_ID}  "
        f"({TIME_START}–{TIME_END})",
        fontsize=13, pad=12
    )

    # ── legend ────────────────────────────────────────────────────────────
    legend_handles = []
    # only show categories that actually appear in the plot
    category_labels = {
        "cat_R-SAD": "Regionale SAD",
        "cat_R-TI":  "Regionale Trenitalia",
        "cat_IRE":   "InterRegioExpress",
    }
    for cat_id, color in CATEGORY_COLORS.items():
        if cat_id in plotted_categories:
            legend_handles.append(
                Line2D([0], [0], color=color, lw=2,
                       label=category_labels.get(cat_id, cat_id))
            )
    if None in plotted_categories or any(
        c not in CATEGORY_COLORS for c in plotted_categories if c is not None
    ):
        legend_handles.append(
            Line2D([0], [0], color=DEFAULT_COLOR, lw=2, label="Other / unknown")
        )

    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right",
                  fontsize=10, framealpha=0.95)

    plt.tight_layout()
    out = "timetable_output.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    root = load_tree(RAILML_FILE)

    print("Step 1: resolving line track references …")
    track_ids = get_line_track_refs(root, LINE_ID)
    print(f"  → {len(track_ids)} tracks: {track_ids}")

    print("Step 2: collecting OCP positions from tracks …")
    ocp_positions = get_ocp_positions(root, track_ids)
    print(f"  → {len(ocp_positions)} OCPs with positions")

    print("Step 3: filtering to passenger OCPs …")
    passenger_ocps = get_passenger_ocps(root, ocp_positions)
    print(f"  → {len(passenger_ocps)} passenger stops on line")
    for ocp_id, (name, km) in sorted(passenger_ocps.items(),
                                      key=lambda x: x[1][1]):
        print(f"     {km:6.2f} km  {name}  ({ocp_id})")

    print("Step 4: parsing train paths …")
    train_paths = get_train_paths(root, LINE_ID, passenger_ocps)
    print(f"  → {len(train_paths)} trainParts found on this line")

    if not train_paths:
        sys.exit("ERROR: No train paths found for this line. "
                 "Check LINE_ID and that sectionTT lineRef values match.")

    print("Step 5: extracting trip data …")
    trip_data = get_trip_data(root, LINE_ID, passenger_ocps)
    print(f"  → extracted {len(trip_data)} trips")
    #delete stations not served
    trip_data = {
    tid: [stops[0]] + [s for s in stops[1:-1] if s[1] != s[2]] + [stops[-1]]
    for tid, stops in trip_data.items()
    }

    print("Step 6: saving trip data …")
    railml_dir = os.path.dirname(os.path.abspath(RAILML_FILE))
    output_path = os.path.join(railml_dir, f"{file_name}.npy")
    np.save(output_path, trip_data, allow_pickle=True)
    print(f"  → saved to {output_path}")

    print("Step 7: plotting …")
    plot_timetable(passenger_ocps, train_paths)


if __name__ == "__main__":
    main()