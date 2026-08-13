import csv
import json
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_CSV = "nodes_copy.csv"
OUTPUT_CSV = "nodes.csv"
INPUT_JSON = "vinschgau.json.txt"

# Lateral displacement corresponding to y = 1.
# Actual displacement = y * SIDE_OFFSET_M
SIDE_OFFSET_M = 5.0

# ---------------------------------------------------------------------------
# Geographic helpers
# ---------------------------------------------------------------------------

EARTH_RADIUS_M = 6371000.0


def latlon_to_xy(lat, lon, lat0, lon0):
    """Convert lat/lon to local metric coordinates around lat0/lon0."""
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    lat0_r = math.radians(lat0)
    lon0_r = math.radians(lon0)

    x = EARTH_RADIUS_M * (lon_r - lon0_r) * math.cos(lat0_r)
    y = EARTH_RADIUS_M * (lat_r - lat0_r)

    return x, y


def xy_to_latlon(x, y, lat0, lon0):
    """Convert local metric coordinates back to lat/lon."""
    lat0_r = math.radians(lat0)
    lon0_r = math.radians(lon0)

    lat = math.degrees(y / EARTH_RADIUS_M + lat0_r)
    lon = math.degrees(x / (EARTH_RADIUS_M * math.cos(lat0_r)) + lon0_r)

    return lat, lon


# ---------------------------------------------------------------------------
# Geometry handling
# ---------------------------------------------------------------------------

def load_main_trace(filename):
    """Load and stitch all relation ways into Meran -> Malles order."""

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    relations = [e for e in data["elements"] if e["type"] == "relation"]

    if not relations:
        raise ValueError("No relation found in OSM JSON.")

    # The supplied file contains the railway relation.
    relation = relations[0]

    ways = []

    for member in relation.get("members", []):
        if member.get("type") != "way":
            continue

        geometry = member.get("geometry")
        if not geometry:
            continue

        ways.append({
            "ref": member["ref"],
            "geometry": geometry
        })

    if not ways:
        raise ValueError("No way geometries found in relation.")

    # -----------------------------------------------------------------------
    # Stitch ways together.
    #
    # OSM way geometries can occur in either direction. We start with the
    # easternmost point because Meran -> Malles corresponds approximately to
    # east -> west in this railway.
    # -----------------------------------------------------------------------

    def point_key(p):
        return (round(p["lat"], 9), round(p["lon"], 9))

    # Start with the way whose geometry contains the easternmost point.
    start_idx = max(
        range(len(ways)),
        key=lambda i: max(p["lon"] for p in ways[i]["geometry"])
    )

    first = ways.pop(start_idx)
    trace = list(first["geometry"])

    while ways:
        current = trace[-1]
        current_key = point_key(current)

        found = None

        for i, way in enumerate(ways):
            geom = way["geometry"]

            if point_key(geom[0]) == current_key:
                found = (i, geom)
                break

            if point_key(geom[-1]) == current_key:
                found = (i, list(reversed(geom)))
                break

        if found is None:
            # Try approximate endpoint matching in case of tiny coordinate
            # differences between OSM ways.
            best = None

            for i, way in enumerate(ways):
                geom = way["geometry"]

                for reversed_order, candidate in [
                    (False, geom),
                    (True, list(reversed(geom)))
                ]:
                    d = (
                        (candidate[0]["lat"] - current["lat"]) ** 2 +
                        (candidate[0]["lon"] - current["lon"]) ** 2
                    )

                    if best is None or d < best[0]:
                        best = (d, i, candidate)

            if best is None:
                raise ValueError("Could not continue stitching OSM ways.")

            # Guard against accidentally connecting unrelated ways.
            if best[0] > 1e-12:
                raise ValueError(
                    f"Could not reliably stitch main trace. "
                    f"Closest endpoint difference: {best[0]}"
                )

            found = (best[1], best[2])

        i, geom = found
        ways.pop(i)

        # Avoid duplicating the shared endpoint.
        trace.extend(geom[1:])

    # Remove consecutive duplicate points.
    cleaned = [trace[0]]

    for p in trace[1:]:
        if point_key(p) != point_key(cleaned[-1]):
            cleaned.append(p)

    return cleaned


def build_metric_trace(trace):
    """Convert trace to local metric coordinates and cumulative distance."""

    lat0 = sum(p["lat"] for p in trace) / len(trace)
    lon0 = sum(p["lon"] for p in trace) / len(trace)

    points = []

    for p in trace:
        x, y = latlon_to_xy(p["lat"], p["lon"], lat0, lon0)
        points.append((x, y))

    cumulative = [0.0]

    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        cumulative.append(
            cumulative[-1] + math.hypot(dx, dy)
        )

    return points, cumulative, lat0, lon0


def interpolate_trace_position(points, cumulative, distance):
    """
    Return position and tangent at a given distance along the trace.

    The distance is clamped to the available geometry.
    """

    if distance <= cumulative[0]:
        i = 0
    elif distance >= cumulative[-1]:
        i = len(points) - 2
    else:
        # Find the segment containing the requested chainage.
        i = 0
        for j in range(len(cumulative) - 1):
            if cumulative[j] <= distance <= cumulative[j + 1]:
                i = j
                break

    x1, y1 = points[i]
    x2, y2 = points[i + 1]

    segment_length = cumulative[i + 1] - cumulative[i]

    if segment_length == 0:
        raise ValueError("Zero-length segment in main trace.")

    t = (distance - cumulative[i]) / segment_length
    t = max(0.0, min(1.0, t))

    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    dx = x2 - x1
    dy = y2 - y1

    length = math.hypot(dx, dy)

    # Tangent pointing Meran -> Malles.
    tx = dx / length
    ty = dy / length

    return x, y, tx, ty


# ---------------------------------------------------------------------------
# CSV handling
# ---------------------------------------------------------------------------

def parse_number(value):
    """Parse numbers using either . or , as decimal separator."""
    if value is None:
        return None

    value = value.strip()

    if not value:
        return None

    return float(value.replace(",", "."))


def format_coordinate(value):
    """Write coordinates using decimal dots."""
    return f"{value:.8f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    base_dir = Path(__file__).resolve().parent

    json_path = base_dir / INPUT_JSON
    csv_path = base_dir / INPUT_CSV
    output_path = base_dir / OUTPUT_CSV

    print(f"Loading main trace: {json_path}")
    trace = load_main_trace(json_path)

    print(f"Main trace contains {len(trace)} geometry points.")

    points, cumulative, lat0, lon0 = build_metric_trace(trace)

    print(
        f"Main trace length: {cumulative[-1]:.1f} m"
    )

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")

        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")

        fieldnames = reader.fieldnames
        rows = list(reader)

    required_fields = {"pk_rel", "y", "coord_x", "coord_y"}

    missing = required_fields - set(fieldnames)

    if missing:
        raise ValueError(
            f"CSV is missing required columns: {', '.join(sorted(missing))}"
        )

    shifted_count = 0

    for row in rows:
        y_value = parse_number(row["y"])

        # Mainline nodes: no lateral displacement.
        if y_value is None or y_value == 0:
            continue

        pk_rel = parse_number(row["pk_rel"])

        if pk_rel is None:
            raise ValueError(
                f"Node {row.get('node_id', '?')} has y != 0 "
                f"but no pk_rel."
            )

        # pk_rel is in km.
        distance_m = pk_rel * 1000.0

        x, y, tx, ty = interpolate_trace_position(
            points,
            cumulative,
            distance_m
        )

        # For a tangent (tx, ty) pointing Meran -> Malles:
        #
        # left  = (-ty, tx)
        # right = ( ty,-tx)
        #
        # Therefore:
        # displacement = y_value * SIDE_OFFSET_M * left_vector
        #
        # This directly implements:
        #   y > 0 -> left
        #   y < 0 -> right

        left_x = -ty
        left_y = tx

        offset = y_value * SIDE_OFFSET_M

        shifted_x = x + offset * left_x
        shifted_y = y + offset * left_y

        lat, lon = xy_to_latlon(
            shifted_x,
            shifted_y,
            lat0,
            lon0
        )

        # CSV convention:
        # coord_x = longitude
        # coord_y = latitude
        row["coord_x"] = format_coordinate(lon)
        row["coord_y"] = format_coordinate(lat)

        shifted_count += 1

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=";",
            lineterminator="\n"
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Shifted {shifted_count} side nodes.")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()