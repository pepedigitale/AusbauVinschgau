import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from pathlib import Path as path
import sys

plt.rcParams["font.family"] = ["Trebuchet MS", "DejaVu Sans"]

project_root = path(r"C:\Users\LeoC\VSCodes\optimizationVinschgau\AusbauVinschgau")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.schematic_map.routing import (build_route,get_signal_nodes_on_route,)
from infra_data.scenarios import get_scenario
scenario = "1a"

trip_data = np.load(rf"C:\Users\LeoC\VSCodes\optimizationVinschgau\AusbauVinschgau\tools\RailML2trip_data\trip_data_{scenario}.npy", allow_pickle=True).item()

# ============================================================
# CONFIGURATION
# ============================================================

CURRENT_STATION = "ME"       # station represented by the ring
CENTER_STATION = "BZ"        # station represented in the center
OUTSIDE_STATION = "MAL"      # station outside the ring

# Background / ring colors
OUTSIDE_BG = "#FFFFFF"
CIRCLE_BG = "#FFFFFF"
RING_COLOR = "#000000"

# Text
TEXT_COLOR = "#000000"
RING_TEXT_COLOR = "#FFFFFF"

# Ring
RING_RADIUS = 1.0
RING_WIDTH = 34
OVERLAY_RING_WIDTH = RING_WIDTH * 0.9
RING_INNER_RADIUS = RING_RADIUS
RING_OUTER_RADIUS = RING_RADIUS

# Arrow configuration
CATEGORY_LENGTH = {
    "fast": 0.70,    # long arrows
    "slow": 0.38,    # short arrows
}

CATEGORY_WIDTH = {
    "fast": 3.5,
    "slow": 2.5,
}

# Easy-to-edit hard-coded colors
CATEGORY_COLOR = {
    "fast": "#000000",
    "slow": "#000000",
}

# Arrow head size
ARROW_HEAD = 18

# Distance from center at which inward arrows terminate
CENTER_ARROW_END = 0.12

# Example timetable
#
# direction:
#   "to_center"   = CURRENT_STATION -> CENTER_STATION
#   "from_center" = CENTER_STATION -> CURRENT_STATION
#   "to_outside"  = CURRENT_STATION -> OUTSIDE_STATION
#   "from_outside"= OUTSIDE_STATION -> CURRENT_STATION
#
# time can be "HH:MM" or simply minutes within the hour.
# ============================================================
# TRAINS
# ============================================================

# Manually defined ME <-> BZ trains
TRAINS = [
    {"time": "00:01", "direction": "to_center",   "category": "fast"},
    {"time": "00:09", "direction": "to_center",   "category": "slow"},
    {"time": "00:20", "direction": "from_center", "category": "slow"},
    {"time": "00:29", "direction": "from_center", "category": "fast"},
    {"time": "00:31", "direction": "to_center",   "category": "fast"},
    {"time": "00:39", "direction": "to_center",   "category": "slow"},
    {"time": "00:50", "direction": "from_center", "category": "slow"},
    {"time": "00:59", "direction": "from_center", "category": "fast"},
]


# Automatically retrieve ME <-> MAL trains from trip_data
seen_minutes = set()

for trip in trip_data.values():
    # Find ME and MAL stops in this trip
    me_stop = next((s for s in trip if s[0] == CURRENT_STATION), None)
    mal_stop = next((s for s in trip if s[0] == OUTSIDE_STATION), None)

    if me_stop is None or mal_stop is None:
        continue

    me_idx = trip.index(me_stop)
    mal_idx = trip.index(mal_stop)

    # Only consider trains that actually run between ME and MAL
    if me_idx < mal_idx:
        # ME -> MAL: use departure from ME
        dt = me_stop[2]
        direction = "to_outside"
    else:
        # MAL -> ME: use arrival/departure appropriately
        dt = me_stop[1]
        direction = "from_outside"

    minute = dt.minute

    # Skip if another loaded train already occupies this minute
    if minute in seen_minutes:
        continue

    seen_minutes.add(minute)

    TRAINS.append({
        "time": dt.strftime("%H:%M"),
        "direction": direction,
        "category": "fast" if len(trip) < 10 else "slow",
    })


# ============================================================
# HELPERS
# ============================================================

def minute_from_time(t):
    """Convert HH:MM into minutes within the hour."""
    return int(t.split(":")[1])


def polar_xy(radius, minute):
    """
    Convert a minute position to x/y.

    :00 is at the top of the circle.
    Time proceeds clockwise:
       :00 top
       :15 right
       :30 bottom
       :45 left
    """
    angle = np.pi / 2 - 2 * np.pi * (minute / 60)
    return radius * np.cos(angle), radius * np.sin(angle)


def tangent_rotation(minute):
    """Return a readable tangent angle for a minute position on the ring."""
    return ((-6 * minute + 90) % 180) - 90


def draw_arrow(ax, train):
    minute = minute_from_time(train["time"])
    direction = train["direction"]
    category = train["category"]

    length = CATEGORY_LENGTH[category]
    width = CATEGORY_WIDTH[category]
    color = CATEGORY_COLOR[category]

    # Position of the train on the ring
    ring_x, ring_y = polar_xy(RING_RADIUS, minute)

    # Unit vector from center -> ring
    distance = np.hypot(ring_x, ring_y)
    ux, uy = ring_x / distance, ring_y / distance

    # --------------------------------------------------------
    # CURRENT -> CENTER
    # Arrow starts at ring and points inward
    # --------------------------------------------------------
    if direction == "to_center":
        start_r = RING_INNER_RADIUS
        end_r = max(CENTER_ARROW_END, RING_RADIUS - length)

        start = (ux * start_r, uy * start_r)
        end = (ux * end_r, uy * end_r)

    # --------------------------------------------------------
    # CENTER -> CURRENT
    # Arrow starts inside and points outward to ring
    # --------------------------------------------------------
    elif direction == "from_center":
        start_r = max(CENTER_ARROW_END, RING_RADIUS - length)
        end_r = RING_INNER_RADIUS

        start = (ux * start_r, uy * start_r)
        end = (ux * end_r, uy * end_r)

    # --------------------------------------------------------
    # CURRENT -> OUTSIDE
    # Arrow starts at ring and points outward
    # --------------------------------------------------------
    elif direction == "to_outside":
        start_r = RING_OUTER_RADIUS
        end_r = RING_RADIUS + length

        start = (ux * start_r, uy * start_r)
        end = (ux * end_r, uy * end_r)

    # --------------------------------------------------------
    # OUTSIDE -> CURRENT
    # Arrow starts outside and points towards ring
    # --------------------------------------------------------
    elif direction == "from_outside":
        start_r = RING_RADIUS + length
        end_r = RING_OUTER_RADIUS

        start = (ux * start_r, uy * start_r)
        end = (ux * end_r, uy * end_r)

    else:
        raise ValueError(f"Unknown direction: {direction}")

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ARROW_HEAD,
        linewidth=width,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=5,
    )

    ax.add_patch(arrow)

    # Put the minute directly on the ring, following its tangent.
    tx, ty = polar_xy(RING_RADIUS, minute)

    ax.text(
        tx,
        ty,
        train["time"].split(":")[1],
        ha="center",
        va="center",
        fontsize=8,
        rotation=tangent_rotation(minute),
        rotation_mode="anchor",
        fontweight="bold",
        color=TEXT_COLOR,
        zorder=10,
    )


# ============================================================
# DRAW
# ============================================================

fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor(OUTSIDE_BG)
ax.set_facecolor(OUTSIDE_BG)

ax.set_xlim(-1.9, 1.9)
ax.set_ylim(-1.9, 1.9)

# Convert the ring's point linewidth into data units so arrow endpoints
# coincide with the actual rendered inner and outer ring edges.
ring_half_width = RING_WIDTH / 72 * fig.dpi / (2 * ax.transData.get_matrix()[0, 0])
RING_INNER_RADIUS = RING_RADIUS - ring_half_width
RING_OUTER_RADIUS = RING_RADIUS + ring_half_width

# Main circle / inner background
circle = Circle(
    (0, 0),
    RING_RADIUS,
    facecolor=CIRCLE_BG,
    edgecolor=RING_COLOR,
    linewidth=RING_WIDTH,
    zorder=1,
)
ax.add_patch(circle)

# ------------------------------------------------------------
# Quarter-hour marks and labels
# ------------------------------------------------------------

for minute, label in [(0, ":00"), (15, ":15"), (30, ":30"), (45, ":45")]:
    # Ticks span the visible ring width and extend slightly beyond its edge.
    x1, y1 = polar_xy(RING_INNER_RADIUS, minute)
    x2, y2 = polar_xy(RING_OUTER_RADIUS + 0.06, minute)

    ax.plot(
        [x1, x2],
        [y1, y2],
        color=RING_COLOR,
        linewidth=5,
        solid_capstyle="round",
        zorder=4,
    )

    # Label
    lx, ly = polar_xy(RING_OUTER_RADIUS + 0.17, minute)

    ax.text(
        lx,
        ly,
        label.removeprefix(":"),
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=TEXT_COLOR,
        zorder=10,
    )

# White inner ring overlay, above the main ring and ticks but below text.
overlay_ring = Circle(
    (0, 0),
    RING_RADIUS,
    facecolor="none",
    edgecolor="#FFFFFF",
    linewidth=OVERLAY_RING_WIDTH,
    zorder=6,
)
ax.add_patch(overlay_ring)

# ------------------------------------------------------------
# Station names
# ------------------------------------------------------------

ax.text(
    0,
    0,
    CENTER_STATION,
    ha="center",
    va="center",
    fontsize=11,
    fontweight="bold",
    color=TEXT_COLOR,
    zorder=10,
)

ax.text(
    0,
    RING_RADIUS + 0.45,
    OUTSIDE_STATION,
    ha="center",
    va="center",
    fontsize=11,
    fontweight="bold",
    color=TEXT_COLOR,
)

# Current station label
ax.text(
    0,
    RING_RADIUS,
    CURRENT_STATION,
    ha="center",
    va="center",
    fontsize=15,
    fontweight="bold",
    color=RING_TEXT_COLOR,
    zorder=11,
)

# ------------------------------------------------------------
# Draw trains
# ------------------------------------------------------------

for train in TRAINS:
    draw_arrow(ax, train)

# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------

'''legend_y = -1.65

for i, category in enumerate(("fast", "slow")):
    y = legend_y - i * 0.14

    ax.plot(
        [-0.75, -0.55],
        [y, y],
        color=CATEGORY_COLOR[category],
        linewidth=CATEGORY_WIDTH[category],
        solid_capstyle="round",
    )

    ax.text(
        -0.48,
        y,
        category,
        ha="left",
        va="center",
        fontsize=9,
        color=TEXT_COLOR,
    )
'''
# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

ax.set_aspect("equal")
ax.axis("off")

fig.suptitle(f"Clock in Meran - scenario {scenario}", fontsize=18, color=TEXT_COLOR)
plt.tight_layout()
output_path = path(__file__).resolve().parent / f"clock_{scenario}.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()