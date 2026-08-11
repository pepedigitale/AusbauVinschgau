from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def canonical_station(station):
    """Convert ME_side -> ME. Stations without '_side' are unchanged."""
    return station[:-5] if station.endswith("_side") else station

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

        for _, d in G.nodes(data=True):
            trains.add(d["train"])
            if d["is_stop"] and d["event"] in {"arr", "dep"}:
                key = (d["train"], d["station"], d["seq"])
                if key not in stop_events:
                    stop_events[key] = {"arr": None, "dep": None}
                stop_events[key][d["event"]] = d["time"]

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

                if delay < 0:
                    station_counts[st]["early"] += 1
                elif delay < 60:
                    station_counts[st]["<60"] += 1
                elif delay < 120:
                    station_counts[st]["60-120"] += 1
                elif delay < 180:
                    station_counts[st]["120-180"] += 1
                elif delay < 240:
                    station_counts[st]["180-240"] += 1
                elif delay < 300:
                    station_counts[st]["240-300"] += 1
                elif delay < 360:
                    station_counts[st]["300-360"] += 1
                else:
                    station_counts[st][">360"] += 1

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

    plt.xticks(x, range(len(scen)))
    plt.xlabel("Scenario")
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

    plt.xticks(x, range(len(scen)))
    plt.xlabel("Scenario")
    plt.ylabel("Total delay [min]")
    plt.title("Delay report")
    plt.grid(axis="y", alpha=.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_punctuality_index(stats):

    order = stats["station_order"]
    counts = stats["stations"]

    early, lt60, r60_120, r120_180, r180_240, r240_300, r300_360, gt360 = [], [], [], [], [], [], [], []

    for st in order:
        c = counts[st]
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
    plt.title("Punctuality index")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=.3)
    plt.legend(title="Delay category (minutes)")
    plt.tight_layout()
    plt.show()