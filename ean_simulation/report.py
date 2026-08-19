from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def canonical_station(station):
    """Convert ME_side -> ME. Stations without '_side' are unchanged."""
    return station[:-5] if station.endswith("_side") else station


def infer_train_direction(G, train_id):
    """Return route direction for a train in a realized graph."""
    relevant = [
        d for _, d in G.nodes(data=True)
        if d.get("train") == train_id and d.get("is_stop")
    ]
    if not relevant:
        return None

    ordered = sorted(relevant, key=lambda x: x["seq"])
    first = canonical_station(ordered[0]["station"])
    last = canonical_station(ordered[-1]["station"])

    if first == "ME" and last == "MAL":
        return "ME->MAL"
    if first == "MAL" and last == "ME":
        return "MAL->ME"
    return None


def infer_train_type(G, train_id):
    """Infer train type from the same fast/slow rule used in Plotly coloring."""
    for module_name in ("visualize_ean_plotly", "ean_simulation.visualize_ean_plotly"):
        try:
            module = __import__(module_name, fromlist=["infer_train_colors"])
            color = module.infer_train_colors(G).get(train_id)
            if color is None:
                continue

            color = (color or "").lower()
            if color in {"#1f77b4", "blue"}:
                return "fast"
            if color in {"#d62728", "#2ca02c", "red", "green"}:
                return "slow"
            return "unknown"
        except Exception:
            continue

    return "unknown"


def train_travel_time(G, train_id):
    """Return travel time in seconds from first seq to last seq for one train."""
    relevant = [
        d for _, d in G.nodes(data=True)
        if d.get("train") == train_id
    ]
    if not relevant:
        return None

    min_seq = min(d.get("seq", 0) for d in relevant)
    max_seq = max(d.get("seq", 0) for d in relevant)

    first_times = [
        float(d["time"])
        for _, d in G.nodes(data=True)
        if d.get("train") == train_id and d.get("seq") == min_seq
    ]
    last_times = [
        float(d["time"])
        for _, d in G.nodes(data=True)
        if d.get("train") == train_id and d.get("seq") == max_seq
    ]

    if not first_times or not last_times:
        return None

    return max(last_times) - min(first_times)


def summarize_travel_time_report(scheduled_graph, realized_graphs):
    """Summarize scheduled and realized travel times by fast/slow train type."""
    trains = sorted({d["train"] for _, d in scheduled_graph.nodes(data=True)})

    scheduled_by_type = defaultdict(list)
    realized_by_type = defaultdict(list)
    train_types = {}
    for train_id in trains:
        train_type = infer_train_type(scheduled_graph, train_id)
        if train_type == "unknown":
            continue

        scheduled_time = train_travel_time(scheduled_graph, train_id)
        if scheduled_time is not None:
            scheduled_by_type[train_type].append((train_id, scheduled_time))

        for G in realized_graphs:
            realized_time = train_travel_time(G, train_id)
            if realized_time is not None:
                realized_by_type[train_type].append(realized_time)

        train_types[train_id] = train_type

    summary = {
        "scheduled": {},
        "realized": {},
    }

    for train_type in sorted(scheduled_by_type):
        values = np.array([v for _, v in scheduled_by_type[train_type]], dtype=float)
        summary["scheduled"][train_type] = {
            "mean_seconds": float(values.mean()),
            "mean_minutes": float(values.mean() / 60.0),
            "n_trains": len(values),
        }

    for train_type in sorted(realized_by_type):
        values = np.array(realized_by_type[train_type], dtype=float)
        summary["realized"][train_type] = {
            "mean_seconds": float(values.mean()),
            "mean_minutes": float(values.mean() / 60.0),
            "n_observations": len(values),
        }

    summary["train_types"] = train_types
    return summary


def print_travel_time_report(report):
    """Print a compact scheduled timetable report by train type."""
    print("Scheduled timetable report")
    print("-" * 72)

    for train_type in sorted(report["scheduled"]):
        scheduled = report["scheduled"][train_type]
        realized = report["realized"].get(train_type, {})

        print(
            f"{train_type:>5} trains: "
            f"scheduled {scheduled['mean_minutes']:.1f} min/train "
            f"({scheduled['n_trains']} trains); "
            f"realized avg {realized.get('mean_minutes', float('nan')):.1f} min"
        )

    print("-" * 72)


FAST_TRAIN_STOPS = ("ME", "NAT", "LAC", "SIL", "LASA", "MAL")


def _stop_distance_km(nodesDf, origin, destination):
    """Distance between two major stops based on pk_rel difference in km."""
    if origin not in nodesDf.index or destination not in nodesDf.index:
        return np.nan
    origin_pk = float(nodesDf.loc[origin, "pk_rel"])
    destination_pk = float(nodesDf.loc[destination, "pk_rel"])
    return abs(destination_pk - origin_pk)


def _ordered_train_stops(G, train_id, stop_names=FAST_TRAIN_STOPS):
    """Return the station sequence for one train, limited to the requested stop set."""
    relevant = []
    for _, d in G.nodes(data=True):
        if d.get("train") != train_id:
            continue
        if not d.get("is_stop"):
            continue
        station = canonical_station(d.get("station", ""))
        if station in stop_names:
            relevant.append({
                "station": station,
                "seq": int(d.get("seq", 0)),
                "event": d.get("event"),
                "time": float(d.get("time", 0.0)),
            })

    relevant.sort(key=lambda x: x["seq"])
    ordered = []
    seen = set()
    for item in relevant:
        if item["station"] not in seen:
            ordered.append(item["station"])
            seen.add(item["station"])
    return ordered


def _station_event_times(G, train_id, stop_names=FAST_TRAIN_STOPS):
    """Map each station to its arrival and departure times in a realized graph."""
    times = defaultdict(dict)
    for _, d in G.nodes(data=True):
        if d.get("train") != train_id:
            continue
        if not d.get("is_stop"):
            continue

        station = canonical_station(d.get("station", ""))
        if station not in stop_names:
            continue

        event = d.get("event")
        if event in {"arr", "dep"}:
            times[station][event] = float(d.get("time", 0.0))

    return dict(times)


def _realized_speed_contributions_for_train(G, train_id, nodesDf, stop_names=FAST_TRAIN_STOPS):
    """Yield (origin, destination, speed_kmh) values for train_id in one realized graph."""
    ordered = _ordered_train_stops(G, train_id, stop_names)
    if len(ordered) < 2:
        return []

    station_times = _station_event_times(G, train_id, stop_names)
    contributions = []

    for i, origin in enumerate(ordered[:-1]):
        for destination in ordered[i + 1:]:
            if origin == destination:
                continue

            origin_events = station_times.get(origin, {})
            dest_events = station_times.get(destination, {})
            dep_time = origin_events.get("dep", origin_events.get("arr"))
            arr_time = dest_events.get("arr", dest_events.get("dep"))

            if dep_time is None or arr_time is None:
                continue

            duration_s = float(arr_time - dep_time)
            if duration_s <= 0:
                continue

            distance_km = _stop_distance_km(nodesDf, origin, destination)
            if not np.isfinite(distance_km) or distance_km <= 0:
                continue

            speed_kmh = (distance_km * 3600.0) / duration_s
            contributions.append((origin, destination, speed_kmh))

    return contributions


def build_speed_matrices(graphs, nodesDf, stop_names=FAST_TRAIN_STOPS):
    """Return (fast_matrix, slow_matrix) with average realized speed in km/h."""
    fast_pairs = defaultdict(list)
    slow_pairs = defaultdict(list)

    for G in graphs:
        trains = sorted({d["train"] for _, d in G.nodes(data=True) if "train" in d})
        for train_id in trains:
            train_type = infer_train_type(G, train_id)
            if train_type not in {"fast", "slow"}:
                continue

            for origin, destination, speed_kmh in _realized_speed_contributions_for_train(
                G, train_id, nodesDf, stop_names
            ):
                bucket = fast_pairs if train_type == "fast" else slow_pairs
                bucket[(origin, destination)].append(float(speed_kmh))

    def matrix_from_pairs(pair_map):
        idx = {station: i for i, station in enumerate(stop_names)}
        matrix = np.full((len(stop_names), len(stop_names)), np.nan)
        for (origin, destination), speeds in pair_map.items():
            if origin not in idx or destination not in idx:
                continue
            matrix[idx[origin], idx[destination]] = float(np.mean(speeds))

        # vertical order: MAL -> ... -> ME, while x-axis remains ME -> ... -> MAL
        reverse_order = list(reversed(stop_names))
        reverse_idx = {station: i for i, station in enumerate(reverse_order)}
        reversed_matrix = np.full((len(stop_names), len(stop_names)), np.nan)
        for origin in stop_names:
            for destination in stop_names:
                value = matrix[idx[origin], idx[destination]]
                reversed_matrix[reverse_idx[origin], idx[destination]] = value
        return reversed_matrix

    return matrix_from_pairs(fast_pairs), matrix_from_pairs(slow_pairs)


def plot_speed_matrix(matrix, labels, title):
    """Plot a single speed matrix with red-to-green color scale and 50% transparency."""
    fig, ax = plt.subplots(figsize=(7, 6))

    cmap = plt.get_cmap("RdYlGn")
    vmin = np.nanmin(matrix) if np.isfinite(matrix).any() else 0.0
    vmax = np.nanmax(matrix) if np.isfinite(matrix).any() else 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.5)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Destination")
    ax.set_ylabel("Origin")
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            ax.text(j, i, f"{value:.1f}", ha="center", va="center",
                    color="black", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Realized speed [km/h]")
    plt.tight_layout()
    plt.show()


def plot_speed_matrices(graphs, nodesDf, stop_names=FAST_TRAIN_STOPS):
    """Create the fast and slow realized-speed matrices side by side."""
    fast_matrix, slow_matrix = build_speed_matrices(graphs, nodesDf, stop_names)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.get_cmap("RdYlGn")

    for ax, matrix, title in [
        (axes[0], fast_matrix, "Fast trains: realized speed [km/h]"),
        (axes[1], slow_matrix, "Slow trains: realized speed [km/h]"),
    ]:
        vmin = np.nanmin(matrix) if np.isfinite(matrix).any() else 0.0
        vmax = np.nanmax(matrix) if np.isfinite(matrix).any() else 1.0
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 1.0

        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.5)

        origin_labels = list(reversed(stop_names))
        ax.set_xticks(range(len(stop_names)))
        ax.set_yticks(range(len(origin_labels)))
        ax.set_xticklabels(stop_names, rotation=45, ha="right")
        ax.set_yticklabels(origin_labels)
        ax.set_xlabel("Destination")
        ax.set_ylabel("Origin")
        ax.set_title(title)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if np.isnan(value):
                    continue
                ax.text(j, i, f"{value:.1f}", ha="center", va="center",
                        color="black", fontsize=8)

    plt.show()

    return fast_matrix, slow_matrix


def extract_statistics(realized_graphs):
    """
    One pass over all realized graphs.

    Returns
    -------
    stats["scenario"] : list of dicts
        Statistics for diagrams 1 & 2.

    stats["stations"] : dict
        Counts for punctuality index.

    stats["station_order"] : list
        Ordered according to train 1.
    """

    scenario_stats = []

    station_counts = defaultdict(lambda: {
        "early": 0,
        "<60": 0,
        "60-120": 0,
        "120-180": 0,
        "180-240": 0,
        "240-300": 0,
        "300-360": 0,
        ">360": 0
    })
    station_counts_by_direction = defaultdict(lambda: defaultdict(lambda: {
        "early": 0,
        "<60": 0,
        "60-120": 0,
        "120-180": 0,
        "180-240": 0,
        "240-300": 0,
        "300-360": 0,
        ">360": 0
    }))

    station_order = None
    valid_stop_count = 0
    theoretical_stop_count = 0

    for scenario_idx, G in enumerate(realized_graphs):

        dep_total = 0
        dep_delayed = 0
        dep_delay_sum = 0.0

        arr_total = 0
        arr_delayed = 0
        arr_delay_sum = 0.0

        last_arrival = {}

        stop_events = {}
        trains = set()
        train_directions = {}

        for _, d in G.nodes(data=True):
            trains.add(d["train"])
            if d["is_stop"] and d["event"] in {"arr", "dep"}:
                key = (d["train"], d["station"], d["seq"])
                if key not in stop_events:
                    stop_events[key] = {"arr": None, "dep": None}
                stop_events[key][d["event"]] = d["time"]

        for train_id in trains:
            train_directions[train_id] = infer_train_direction(G, train_id)

        valid_stop = {}
        for key, times in stop_events.items():
            arr_time = times["arr"]
            dep_time = times["dep"]
            if arr_time is not None and dep_time is not None:
                valid_stop[key] = (dep_time - arr_time) >= 1
            else:
                valid_stop[key] = True

        if station_order is None:
            train1 = [
                d for _, d in G.nodes(data=True)
                if d["train"] == 1 and d["is_stop"]
            ]
            train1.sort(key=lambda x: x["seq"])

            station_order = []
            seen = set()

            for d in train1:
                st = canonical_station(d["station"])
                if st not in seen:
                    station_order.append(st)
                    seen.add(st)

        theoretical_stop_count += len(trains) * len(station_order)
        valid_stop_count += sum(valid_stop.values())

        for _, d in G.nodes(data=True):

            delay = d["time"] - d["scheduled_time"]
            stop_key = (d["train"], d["station"], d["seq"])
            is_valid_stop = valid_stop.get(stop_key, True)

            # -------------------------------------------------------
            # Diagram 3
            # -------------------------------------------------------
            if d["is_stop"] and is_valid_stop:
                st = canonical_station(d["station"])
                direction = train_directions.get(d["train"])

                if delay < 0:
                    station_counts[st]["early"] += 1
                    if direction is not None:
                        station_counts_by_direction[direction][st]["early"] += 1
                elif delay < 60:
                    station_counts[st]["<60"] += 1
                    if direction is not None:
                        station_counts_by_direction[direction][st]["<60"] += 1
                elif delay < 120:
                    station_counts[st]["60-120"] += 1
                    if direction is not None:
                        station_counts_by_direction[direction][st]["60-120"] += 1
                elif delay < 180:
                    station_counts[st]["120-180"] += 1
                    if direction is not None:
                        station_counts_by_direction[direction][st]["120-180"] += 1
                elif delay < 240:
                    station_counts[st]["180-240"] += 1
                    if direction is not None:
                        station_counts_by_direction[direction][st]["180-240"] += 1
                elif delay < 300:
                    station_counts[st]["240-300"] += 1
                    if direction is not None:
                        station_counts_by_direction[direction][st]["240-300"] += 1
                elif delay < 360:
                    station_counts[st]["300-360"] += 1
                    if direction is not None:
                        station_counts_by_direction[direction][st]["300-360"] += 1
                else:
                    station_counts[st][">360"] += 1
                    if direction is not None:
                        station_counts_by_direction[direction][st][">360"] += 1

            # -------------------------------------------------------
            # Diagram 1 & 2 (departure)
            # -------------------------------------------------------
            if d["event"] == "dep" and d["seq"] == 0:

                dep_total += 1
                dep_delay_sum += delay

                if delay > 180:
                    dep_delayed += 1

            # -------------------------------------------------------
            # Keep only final arrival
            # -------------------------------------------------------
            if d["event"] == "arr":

                train = d["train"]

                if (
                    train not in last_arrival
                    or d["seq"] > last_arrival[train]["seq"]
                ):
                    last_arrival[train] = d

        # Final arrivals
        for d in last_arrival.values():

            delay = d["time"] - d["scheduled_time"]

            arr_total += 1
            arr_delay_sum += delay

            if delay > 180:
                arr_delayed += 1

        scenario_stats.append({
            "dep_delayed_pct": 100 * dep_delayed / dep_total,
            "arr_delayed_pct": 100 * arr_delayed / arr_total,
            "dep_delay_sum_min": dep_delay_sum / 60,
            "arr_delay_sum_min": arr_delay_sum / 60,
        })

    return {
        "scenario": scenario_stats,
        "stations": station_counts,
        "stations_by_direction": dict(
            (direction, dict(stations))
            for direction, stations in station_counts_by_direction.items()
        ),
        "station_order": station_order,
        "valid_stop_count": valid_stop_count,
        "theoretical_stop_count": theoretical_stop_count,
    }



def plot_train_report(stats):

    scen = stats["scenario"]

    dep = [s["dep_delayed_pct"] for s in scen]
    arr = [s["arr_delayed_pct"] for s in scen]

    x = np.arange(len(scen))
    w = 0.38

    plt.figure(figsize=(12,5))

    plt.bar(x-w/2, dep, width=w, color="steelblue",
            label="Departure delay >3 min")

    plt.bar(x+w/2, arr, width=w, color="firebrick",
            label="Arrival delay >3 min")

    plt.xticks(x, range(1, len(scen) + 1), fontsize=5)
    plt.xlabel("Realization")
    plt.ylabel("% of trains")
    plt.title("Train report")
    plt.legend()
    plt.grid(axis="y", alpha=.3)

    plt.tight_layout()
    plt.show()


def plot_delay_report(stats):

    scen = stats["scenario"]

    dep = [s["dep_delay_sum_min"] for s in scen]
    arr = [s["arr_delay_sum_min"] for s in scen]

    x = np.arange(len(scen))
    w = 0.38

    plt.figure(figsize=(12,5))

    plt.bar(x-w/2, dep, width=w,
            color="gold",
            label="Departure")

    plt.bar(x+w/2, arr, width=w,
            color="forestgreen",
            label="Arrival")

    plt.xticks(x, range(1, len(scen) + 1), fontsize=5)
    plt.xlabel("Realization")
    plt.ylabel("Total delay [min]")
    plt.title("Delay report")
    plt.grid(axis="y", alpha=.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


def _plot_punctuality_index_counts(order, counts, title):
    early, lt60, r60_120, r120_180, r180_240, r240_300, r300_360, gt360 = [], [], [], [], [], [], [], []

    for st in order:
        c = counts.get(st, {"early": 0, "<60": 0, "60-120": 0, "120-180": 0, "180-240": 0, "240-300": 0, "300-360": 0, ">360": 0})
        total = sum(c.values()) or 1

        early.append(100*c["early"]/total)
        lt60.append(100*c["<60"]/total)
        r60_120.append(100*c["60-120"]/total)
        r120_180.append(100*c["120-180"]/total)
        r180_240.append(100*c["180-240"]/total)
        r240_300.append(100*c["240-300"]/total)
        r300_360.append(100*c["300-360"]/total)
        gt360.append(100*c[">360"]/total)

    x = np.arange(len(order))

    plt.figure(figsize=(15,6))

    b1 = np.array(early)
    b2 = b1 + np.array(lt60)
    b3 = b2 + np.array(r60_120)
    b4 = b3 + np.array(r120_180)
    b5 = b4 + np.array(r180_240)
    b6 = b5 + np.array(r240_300)
    b7 = b6 + np.array(r300_360)

    plt.bar(x, early,      bottom=None, color="forestgreen",  label="Early (delay < 0 s)")
    plt.bar(x, lt60,       bottom=b1,   color="gold",         label="0–1 min")
    plt.bar(x, r60_120,    bottom=b2,   color="darkorange",   label="1–2 min")
    plt.bar(x, r120_180,   bottom=b3,   color="orangered",    label="2–3 min")
    plt.bar(x, r180_240,   bottom=b4,   color="firebrick",    label="3–4 min")
    plt.bar(x, r240_300,   bottom=b5,   color="darkmagenta",  label="4–5 min")
    plt.bar(x, r300_360,   bottom=b6,   color="rebeccapurple", label="5–6 min")
    plt.bar(x, gt360,      bottom=b7,   color="indigo",       label=">6 min")

    plt.xticks(x, order, rotation=45, ha="right")
    plt.ylabel("Share of train observations [%]")
    plt.xlabel("Station")
    plt.title(title)
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=.3)
    plt.legend(title="Delay category (minutes)")
    plt.tight_layout()
    plt.show()


def plot_punctuality_index(stats):
    order = stats["station_order"]
    counts = stats["stations"]
    _plot_punctuality_index_counts(order, counts, "Punctuality index (both directions)")


def plot_punctuality_index_me_to_mal(stats):
    order = stats["station_order"]
    counts = stats["stations_by_direction"].get("ME->MAL", {})
    _plot_punctuality_index_counts(order, counts, "Punctuality index (ME->MAL)")


def plot_punctuality_index_mal_to_me(stats):
    order = stats["station_order"]
    counts = stats["stations_by_direction"].get("MAL->ME", {})
    _plot_punctuality_index_counts(order, counts, "Punctuality index (MAL->ME)")


# Backward-compatible aliases using the requested diagram numbering.
plot_punctuality_index_diagram4 = plot_punctuality_index_me_to_mal
plot_punctuality_index_diagram5 = plot_punctuality_index_mal_to_me