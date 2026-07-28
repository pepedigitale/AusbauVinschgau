import json
import os
import folium
import csv
from pyproj import Geod

# =============================================================================
# CONFIG
# =============================================================================
FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vinschgau.json.txt")

COLOR_SINGLE   = "#c0392b"   # rosso  – binario semplice
COLOR_EXISTING = "#08e01d"   # verde scuro – raddoppio esistente
COLOR_NEW      = "#91f59b"   # verde chiaro – raddoppio futuro (scenario)
LINE_WEIGHT    = 5

# =============================================================================
# LOAD & BUILD ORDERED COORDS
# =============================================================================
with open(FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

relation = next(el for el in data["elements"] if el["type"] == "relation")

ordered_coords = []
for member in relation["members"]:
    if member["type"] == "way" and "geometry" in member:
        coords = [(pt["lon"], pt["lat"]) for pt in member["geometry"]]
        if not coords:
            continue
        if ordered_coords:
            if ordered_coords[-1] == coords[0]:
                ordered_coords.extend(coords[1:])
            else:
                ordered_coords.extend(coords)
        else:
            ordered_coords.extend(coords)

print(f"Punti polyline: {len(ordered_coords)}")

# =============================================================================
# CUMULATIVE DISTANCES  (metres from Merano = point 0)
# =============================================================================
geod = Geod(ellps="WGS84")

cumulative_distances = [0.0]
for i in range(1, len(ordered_coords)):
    lon1, lat1 = ordered_coords[i - 1]
    lon2, lat2 = ordered_coords[i]
    _, _, dist = geod.inv(lon1, lat1, lon2, lat2)
    cumulative_distances.append(cumulative_distances[-1] + dist)

total_length_m = cumulative_distances[-1]
print(f"Lunghezza totale: {total_length_m / 1000:.3f} km")

# =============================================================================
# HELPERS
# =============================================================================

def coord_at_pk(pk_km):
    """Restituisce (lat, lon) interpolando lungo la polyline alla PK relativa."""
    target_m = pk_km * 1000.0
    if target_m <= 0:
        lon, lat = ordered_coords[0]; return lat, lon
    if target_m >= total_length_m:
        lon, lat = ordered_coords[-1]; return lat, lon

    lo, hi = 0, len(cumulative_distances) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if cumulative_distances[mid] <= target_m:
            lo = mid
        else:
            hi = mid

    seg_len = cumulative_distances[hi] - cumulative_distances[lo]
    if seg_len < 1e-9:
        lon, lat = ordered_coords[lo]; return lat, lon

    t = (target_m - cumulative_distances[lo]) / seg_len
    lon1, lat1 = ordered_coords[lo]
    lon2, lat2 = ordered_coords[hi]
    return lat1 + t * (lat2 - lat1), lon1 + t * (lon2 - lon1)


def segment_coords_between(pk_start_km, pk_end_km):
    """
    Restituisce la lista di coordinate [(lat,lon),...] della polyline
    compresa tra pk_start_km e pk_end_km (PK relative).
    """
    s = pk_start_km * 1000.0
    e = pk_end_km   * 1000.0
    result = [coord_at_pk(pk_start_km)]

    for i in range(len(ordered_coords)):
        d = cumulative_distances[i]
        if s < d < e:
            lon, lat = ordered_coords[i]
            result.append((lat, lon))

    result.append(coord_at_pk(pk_end_km))
    return result


def add_colored_line(feature_group, pk_start, pk_end, color,
                     weight=LINE_WEIGHT, opacity=0.95, tooltip=""):
    coords = segment_coords_between(pk_start, pk_end)
    folium.PolyLine(
        coords,
        color=color,
        weight=weight,
        opacity=opacity,
        tooltip=tooltip,
    ).add_to(feature_group)
    return pk_end - pk_start   # km


# =============================================================================
# SIGNAL TABLE: load from CSV `nodes_mainline.csv`
# The CSV uses semicolon separators and commas for decimal points.
# We only import rows where the `y` attribute == 0
# Resulting `points` is a list of tuples: (name, pk_abs, pk_rel, attrs_dict)
# =============================================================================
NODES_CSV_PATH = r"C:\Users\LeoC\VSCodes\optimizationVinschgau\SimulatedAnnealing\infrastructuredesign\map\nodes_mainline.csv"


points = []
with open(NODES_CSV_PATH, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';')
    for row in reader:
        # parse numeric fields that use comma as decimal separator
        y_str = (row.get('y') or '').strip().replace(',', '.')
        if y_str == '':
            continue
        try:
            y_val = float(y_str)
        except ValueError:
            continue
        # only import rows with y == 0
        if abs(y_val) > 1e-9:
            continue

        pk_rel_str = (row.get('pk_rel') or '').strip().replace(',', '.')
        pk_abs_str = (row.get('pk_abs') or '').strip().replace(',', '.')
        try:
            pk_rel = float(pk_rel_str)
            pk_abs = float(pk_abs_str)
        except ValueError:
            continue

        name = (row.get('name') or '').strip()
        attrs = {
            'node_id': (row.get('node_id') or '').strip(),
            'valid_dir': (row.get('valid_dir') or '').strip(),
            'node_type': (row.get('node_type') or '').strip(),
            'stop_slow': (row.get('stop_slow') or '').strip(),
            'stop_fast': (row.get('stop_fast') or '').strip(),
            'y': y_val,
            'altitude': (row.get('altitude') or '').strip().replace(',', '.')
        }

        points.append((name, pk_abs, pk_rel, attrs))

# Lookup nome → pk_relativa  (prima occorrenza)
pk_by_name = {}
for name, _abs, pk_rel, _attrs in points:
    if name not in pk_by_name:
        pk_by_name[name] = pk_rel - 0.025

# =============================================================================
# RADDOPPI ESISTENTI  ±100 m attorno a ogni LdS
# =============================================================================
EXISTING_LDS = ["Merano", "Marlengo", "Tel", "Stava", "Laces",
                "Silandro", "Lasa", "Spondigna", "Malles"]
OFFSET_KM = 0.100

existing_intervals = []
for lds in EXISTING_LDS:
    pk = pk_by_name[lds]
    lo = max(0.0, pk - OFFSET_KM)
    hi = min(total_length_m / 1000.0, pk + OFFSET_KM)
    existing_intervals.append((lo, hi, lds))

# =============================================================================
# SCENARI FUTURI
# =============================================================================
SCENARIOS = {
    "Scenario 1a": [
        ("Tel",          "Plaus"),
        ("SblNatKomp",      "Stava"),
        ("SblStab21529", "SblStab24529"),
        ("Coldrano",     "SblLac35026"),
        ("Lasa",         "Oris"),
    ],

    "Scenario 1b": [
        ("Tel",          "Plaus"),
        ("SblStab21529", "SblStab24529"),
        ("Laces", "Coldrano"),
        ("Lasa",         "Oris"),
    ],

    "Scenario 1c": [
        ("Merano",  "Lagundo"),
        ("Tel",          "Plaus"),
        ("PL10",          "Naturno"),
        ("Stava", "SblStab24529"),
        ("Laces",     "Coldrano"),
        ("Lasa",         "Oris"),
    ],

    "Scenario 2": [
        ("Merano",       "Lagundo"),
        ("Tel",          "Plaus"),
        ("SblStab21529", "SblStab24529"),
        ("SblLac31026",  "SblLac35026"),
        ("Lasa",         "SblLasa46915"),
        ("SblSpon54860",  "Sluderno"),
        ("SblSpon58860", "Malles"),
    ],
    "Scenario 3": [
        ("Tel",      "Plaus"),
        ("Ciardes",  "Castelbello"),
        ("Coldrano", "SblLac35026"),
        ("Oris",     "Spondigna"),
    ],
    "Gallerie": [
        ("Imbocco Galleria di Marlengo",      "Sbocco Galleria di Marlengo"),
        ("Imbocco Galleria Monte Giuseppe",  "Sbocco Galleria Monte Giuseppe"),
        ("Imbocco Galleria artificiale", "Sbocco Galleria artificiale"),
        ("Imbocco Galleria di Tel", "Sbocco Galleria di Tel"),
        ("Imbocco Galleria GEOS",     "Sbocco Galleria GEOS"),
    ],
}


def simplify_tunnel_label(name):
    if name.startswith("Imbocco "):
        return name.removeprefix("Imbocco ")
    if name.startswith("Sbocco "):
        return name.removeprefix("Sbocco ")
    return name

scenario_intervals = {}
for sc_name, pairs in SCENARIOS.items():
    intervals = []
    for a, b in pairs:
        pk_a = pk_by_name[a]
        pk_b = pk_by_name[b]
        if pk_a > pk_b:
            pk_a, pk_b = pk_b, pk_a
        if sc_name == "Gallerie":
            label_a = simplify_tunnel_label(a)
            label_b = simplify_tunnel_label(b)
            label = label_a if label_a == label_b else f"{label_a} → {label_b}"
        else:
            label = f"{a} → {b}"
        intervals.append((pk_a, pk_b, label))
    scenario_intervals[sc_name] = intervals

print("\nLunghezze raddoppi nuovi per scenario:")
for sc_name, intervals in scenario_intervals.items():
    total = sum(b - a for a, b, _ in intervals)
    print(f"  {sc_name}: {total:.3f} km")

# =============================================================================
# NODI COSTO
# =============================================================================
# Estrai stazioni (stop_slow == 1) e nodi Sbl
cost_nodes = []
for name, pk_abs, pk_rel, attrs in points:
    stop_slow = attrs.get('stop_slow', '').strip()
    node_type = attrs.get('node_type', '').strip()
    
    # Includi stazioni (stop_slow == 1) e nodi Sbl
    if (stop_slow == '1') or (node_type == 'Sbl'):
        cost_nodes.append((name, pk_rel))

# Ordina per pk_rel per ottenere l'ordine corretto lungo la linea
cost_nodes.sort(key=lambda x: x[1])

# Crea un mapping nome → pk_relativa per i nodi costo
cost_pk_by_name = {}
for name, pk_rel in cost_nodes:
    if name not in cost_pk_by_name:
        cost_pk_by_name[name] = pk_rel

# ========== DUMMY ARRAY - MODIFICA QUI I COSTI ==========
# Formato: (nodo_partenza, nodo_arrivo, costo per km)
# Modifica i costi secondo le tue necessità
COST_SEGMENTS = [
    ("Merano", "Lagundo", 44),
    ("Lagundo", "Marlengo", 121),
    ("Marlengo", "Tel", 160),
    ("Tel", "Rabla", 34),
    ("Rabla", "Plaus", 56),
    ("Plaus", "Naturno", 78),
    ("Naturno", "SblNatKomp", 145),
    ("SblNatKomp", "Stava", 23),
    ("Stava", "SblStab21529", 138),
    ("SblStab21529", "Ciardes", 24),
    ("Ciardes", "SblStab24529", 12),
    ("SblStab24529", "Castelbello", 58),
    ("Castelbello", "Laces", 135),
    ("Laces", "SblLac31026", 114),
    ("SblLac31026", "Coldrano", 111),
    ("Coldrano", "SblLac35026", 32),
    ("SblLac35026", "Silandro", 141),
    ("Silandro", "Lasa", 87),
    ("Lasa", "SblLasa46915", 122),
    ("SblLasa46915", "Oris", 14),
    ("Oris", "Spondigna", 19),
    ("Spondigna", "SblSpon54860", 11),
    ("SblSpon54860", "SblSpon58860", 97),
    ("SblSpon58860", "Malles", 100),
]


# ========== COST INTERVALS AND SCENARIO COST CALCULATION ==========

def build_cost_intervals(cost_segments, name_to_pk):
    intervals = []
    for start_node, end_node, unit_cost in cost_segments:
        pk_start = name_to_pk.get(start_node)
        pk_end = name_to_pk.get(end_node)
        if pk_start is None or pk_end is None:
            continue
        if pk_start > pk_end:
            pk_start, pk_end = pk_end, pk_start
        intervals.append((pk_start, pk_end, unit_cost))
    intervals.sort(key=lambda item: item[0])
    return intervals


def interval_overlap(lo1, hi1, lo2, hi2):
    return max(0.0, min(hi1, hi2) - max(lo1, lo2))


def compute_scenario_costs(scenario_intervals, cost_intervals):
    results = {}
    for scenario_name, intervals in scenario_intervals.items():
        total_km = 0.0
        total_cost = 0.0
        uncovered_km = 0.0

        for lo, hi, _label in intervals:
            length = max(0.0, hi - lo)
            total_km += length
            covered = 0.0
            for cost_lo, cost_hi, unit_cost in cost_intervals:
                overlap = interval_overlap(lo, hi, cost_lo, cost_hi)
                if overlap > 0:
                    total_cost += overlap * unit_cost
                    covered += overlap
            uncovered_km += max(0.0, length - covered)

        results[scenario_name] = {
            "km": total_km,
            "cost": total_cost,
            "uncovered_km": uncovered_km,
        }
    return results


COST_INTERVALS = build_cost_intervals(COST_SEGMENTS, cost_pk_by_name)
scenario_costs = compute_scenario_costs(scenario_intervals, COST_INTERVALS)

print(f"\nNodi costo identificati: {len(cost_nodes)}")
print(f"Segmenti costo creati: {len(COST_SEGMENTS)}")
print("\nCosti stimati per scenario su base costo unitario e km raddoppiati:")
for sc_name, values in scenario_costs.items():
    print(f"  {sc_name}: {values['km']:.3f} km, costo totale €{values['cost']:.2f}")
    if values['uncovered_km'] > 0:
        print(f"    Attenzione: {values['uncovered_km']:.3f} km non coperti da segmenti costo noti")

# =============================================================================
# FUNZIONE PER MAPPARE COSTO A COLORE
# =============================================================================
def hex_to_rgb(hex_color):
    """Converte colore hex a RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(r, g, b):
    """Converte RGB a colore hex."""
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

def interpolate_color(color1, color2, t):
    """Interpola linearmente tra due colori hex. t è un valore tra 0 e 1."""
    r1, g1, b1 = hex_to_rgb(color1)
    r2, g2, b2 = hex_to_rgb(color2)
    r = r1 + (r2 - r1) * t
    g = g1 + (g2 - g1) * t
    b = b1 + (b2 - b1) * t
    return rgb_to_hex(r, g, b)

def get_color_for_cost(cost):
    """Mappa il costo a un colore usando una scala continua.
    Scala: Verde (basso) -> Arancione -> Rosso -> Rosso scuro (alto)
    """
    if cost <= 50:
        # Verde a Arancione (0-50)
        t = cost / 50
        return interpolate_color("#27ae60", "#f39c12", t)
    elif cost <= 100:
        # Arancione a Rosso (50-100)
        t = (cost - 50) / 50
        return interpolate_color("#f39c12", "#e74c3c", t)
    elif cost <= 150:
        # Rosso a Rosso scuro (100-150)
        t = (cost - 100) / 50
        return interpolate_color("#e74c3c", "#8b0000", t)
    else:
        # Oltre 150, rimani rosso scuro
        return "#8b0000"

# =============================================================================
# BUILD SCENARIO LAYERS
# =============================================================================

def build_scenario_layers(scenario_name, m_map, show=True):
    new_intervals = scenario_intervals.get(scenario_name, [])
    fg_new = folium.FeatureGroup(name=f"🟢 Raddoppi {scenario_name}", show=show)

    for lo, hi, label in new_intervals:
        length_km = hi - lo
        add_colored_line(fg_new, lo, hi, color=COLOR_NEW,
                         tooltip=f"🟢 Raddoppio nuovo: {label} ({length_km:.3f} km)")

    fg_new.add_to(m_map)

def build_tunnel_layer(m_map, show=True):
    tunnel_intervals = scenario_intervals.get("Gallerie", [])
    fg_tunnel = folium.FeatureGroup(name="🕳️ Gallerie", show=show)

    for lo, hi, label in tunnel_intervals:
        length_km = hi - lo
        add_colored_line(fg_tunnel, lo, hi, color="#34495e",
                         tooltip=f"🕳️ {label} ({length_km:.3f} km)")

    fg_tunnel.add_to(m_map)

def build_cost_layer(m_map, show=False):
    """Costruisce layer dei costi segmentati per nodo."""
    fg_cost = folium.FeatureGroup(name="💰 Costi segmenti", show=show)
    
    for start_node, end_node, cost in COST_SEGMENTS:
        pk_start = cost_pk_by_name.get(start_node)
        pk_end = cost_pk_by_name.get(end_node)
        
        if pk_start is None or pk_end is None:
            continue
        
        color = get_color_for_cost(cost)
        label = f"{start_node} → {end_node}"
        length_km = abs(pk_end - pk_start)
        
        add_colored_line(fg_cost, pk_start, pk_end, color=color,
                         tooltip=f"💰 {label}: costo {cost} ({length_km:.3f} km)")
    
    fg_cost.add_to(m_map)

# =============================================================================
# MAPPA
# =============================================================================
center_lat = sum(lat for _, lat in ordered_coords) / len(ordered_coords)
center_lon = sum(lon for lon, _ in ordered_coords) / len(ordered_coords)

m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")
map_name = m.get_name()

#m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="ESRI World Imagery")

#m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="ESRI Topo")

#m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="Stadia Terrain")

#m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB Voyager")

# Layer unificati
layer_binario_semplice    = folium.FeatureGroup(name="🔴 Binario semplice", show=True)
layer_raddoppi_esistenti  = folium.FeatureGroup(name="🟩 Raddoppi esistenti", show=True)

# Linea rossa unica su tutta la tratta
add_colored_line(layer_binario_semplice, 0.0, total_length_m / 1000.0,
                 color=COLOR_SINGLE,
                 tooltip="Binario semplice – tutta la tratta")

# Raddoppi esistenti unificati in un unico layer
for lo, hi, lds in existing_intervals:
    add_colored_line(layer_raddoppi_esistenti, lo, hi,
                     color=COLOR_EXISTING,
                     tooltip=f"🟩 Raddoppio esistente: {lds} (±100 m)")

layer_binario_semplice.add_to(m)
layer_raddoppi_esistenti.add_to(m)

# Scenari – solo Scenario 1 visibile di default
build_scenario_layers("Scenario 1a", m, show=True)
build_scenario_layers("Scenario 1b", m, show=False)
build_scenario_layers("Scenario 1c", m, show=False)
build_scenario_layers("Scenario 2", m, show=False)
build_scenario_layers("Scenario 3", m, show=False)
build_tunnel_layer(m, show=False)
build_cost_layer(m, show=False)

# =============================================================================
# SCENARIO SUMMARY PANEL
# =============================================================================
summary_data = {
    name: {"km": values["km"], "cost": values["cost"]}
    for name, values in scenario_costs.items()
}
summary_json = json.dumps(summary_data, ensure_ascii=False)
summary_layer_names = {
    f"🟢 Raddoppi {name}": name
    for name in scenario_intervals.keys()
}
summary_mapping_json = json.dumps(summary_layer_names, ensure_ascii=False)
summary_panel_html = """
<div id="scenario-summary-panel" style="position:fixed; bottom:20px; right:20px; z-index:1000; max-width:360px; background:rgba(255,255,255,0.96); border:1px solid #2c3e50; border-radius:10px; padding:10px 14px; font-family:sans-serif; font-size:13px; color:#1a1a1a; box-shadow:0 2px 10px rgba(0,0,0,0.28);">
  <strong style="display:block; margin-bottom:8px;">Scenario totals</strong>
  <div id="scenario-summary-content" style="max-height:220px; overflow-y:auto;"> <em>Enable scenario layer(s) to see km and cost totals.</em></div>
</div>
<script>
(function() {{
    var summaryData = {summary_json};
    var labelToScenario = {summary_mapping_json};
    var content = document.getElementById('scenario-summary-content');

    function normalizeText(text) {{
        return text.replace(/\s+/g, ' ').trim();
    }}

    function getCheckedScenarios() {{
        var selected = [];
        var inputs = document.querySelectorAll('.leaflet-control-layers-list input[type="checkbox"]');
        inputs.forEach(function(input) {{
            if (!input.checked) {{
                return;
            }}
            var label = input.parentElement;
            if (!label) {{
                return;
            }}
            var text = normalizeText(label.textContent || label.innerText || '');

            Object.keys(labelToScenario).forEach(function(displayName) {{
                var normalized = normalizeText(displayName);
                if (text === normalized || text.indexOf(normalized) !== -1) {{
                    selected.push(labelToScenario[displayName]);
                }}
            }});
        }});
        return selected;
    }}

    function render() {{
        if (!content) {{
            return;
        }}
        var visible = getCheckedScenarios();
        if (visible.length === 0) {{
            content.innerHTML = '<em>Enable scenario layer(s) to see km and cost totals.</em>';
            return;
        }}
        visible.sort();
        var lines = ['<ul style="margin:0; padding:0 0 0 18px;">'];
        visible.forEach(function(scenarioName) {{
            var values = summaryData[scenarioName];
            if (!values) return;
            lines.push('<li style="margin-bottom:4px; line-height:1.4;">' + scenarioName + ': ' + values.km.toFixed(3) + ' km, €' + values.cost.toFixed(2) + '</li>');
        }});
        lines.push('</ul>');
        content.innerHTML = lines.join('');
    }}

    function attachHandlers() {{
        var inputs = document.querySelectorAll('.leaflet-control-layers-list input[type="checkbox"]');
        inputs.forEach(function(input) {{
            input.addEventListener('change', render);
        }});
    }}

    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', function() {{
            attachHandlers();
            render();
        }});
    }} else {{
        attachHandlers();
        render();
    }}
}})();
</script>
""".format(summary_json=summary_json, summary_mapping_json=summary_mapping_json, map_name=map_name)

m.get_root().html.add_child(folium.Element(summary_panel_html))

# =============================================================================
# MARKER SEGNALI
# =============================================================================
STATIONS = {
    "Merano", "Lagundo", "Marlengo", "Tel", "Rabla", "Plaus",
    "Naturno", "Stava", "Ciardes", "Castelbello", "Laces",
    "Coldrano", "Silandro", "Lasa", "Oris", "Spondigna", "Sluderno", "Malles"
}

def classify(name):
    if "Prot." in name:
        return "Protezione"
    if "Part." in name:
        return "Partenza"
    if "PL" in name:
        return "PL"
    if "HD" in name:
        return "HD"
    if "Sbl" in name:
        return "Sbl"
    if name in STATIONS:
        return "LdS"

    return "other"


STYLE = {
    "LdS":        dict(color="#1a1a2e", icon="train", prefix="fa"),
    "PL":         dict(color="#d35400", icon="times", prefix="fa", radius=5),
    "HD":         dict(color="#ff00ff", icon="bolt", prefix="fa", radius=6),
    "Sbl":        dict(color="#f39c12", icon="circle", prefix="fa", radius=5),
    "Partenza":   dict(color="#27ae60", icon="arrow-right", prefix="fa", radius=5),
    "Protezione": dict(color="#c0392b", icon="shield", prefix="fa", radius=5),
    "other":      dict(color="#7f8c8d", icon="info-circle", prefix="fa", radius=4),
}


# --- LAYERS ---
layer_lds        = folium.FeatureGroup(name="🚉 LdS (Località di servizio)", show=True)
layer_pl         = folium.FeatureGroup(name="🚧 Passaggi a Livello", show=False)
layer_hd         = folium.FeatureGroup(name="⚡ Segnali HD", show=False)
layer_sbl        = folium.FeatureGroup(name="🔶 Segnali Sbl", show=False)
layer_partenza   = folium.FeatureGroup(name="🟢 Segnali di Partenza", show=False)
layer_protezione = folium.FeatureGroup(name="🔴 Segnali di Protezione", show=False)
layer_other      = folium.FeatureGroup(name="ℹ️ Altri punti", show=False)


LAYERS = {
    "LdS": layer_lds,
    "PL": layer_pl,
    "HD": layer_hd,
    "Sbl": layer_sbl,
    "Partenza": layer_partenza,
    "Protezione": layer_protezione,
    "other": layer_other,
}


# --- LOOP ---
for name, pk_abs, pk_rel, attrs in points:
    lat, lon = coord_at_pk(pk_rel - 0.025)

    cat   = classify(name)
    style = STYLE[cat]
    layer = LAYERS[cat]

    popup_html = (
        f'<div style="font-family:monospace;min-width:200px">'
        f'<b style="font-size:13px">{name}</b><br>'
        f'<span style="color:#555">ID:</span> <b>{attrs.get("node_id","")}</b><br>'
        f'<span style="color:#555">PK assoluta:</span> <b>{pk_abs:.3f} km</b><br>'
        f'<span style="color:#555">PK relativa:</span> <b>{pk_rel:.3f} km</b><br>'
        f'<span style="color:#555">Tipo nodo:</span> <b>{attrs.get("node_type","")}</b><br>'
        f'<span style="color:#555">Direzione:</span> <b>{attrs.get("valid_dir","")}</b><br>'
        f'<span style="color:#555">Fermata lenta/veloce:</span> <b>{attrs.get("stop_slow","")}/{attrs.get("stop_fast","")}</b><br>'
        f'<span style="color:#555">Altitudine:</span> <b>{attrs.get("altitude","")}</b>'
        f'</div>'
    )

    if cat == "LdS":
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"🚉 {name} (PK {pk_abs:.3f})",
            icon=folium.Icon(
                color="darkblue",
                icon_color="white",
                icon=style["icon"],
                prefix=style["prefix"]
            )
        ).add_to(layer)
    else:
        folium.CircleMarker(
            location=[lat, lon],
            radius=style.get("radius", 4),
            color=style["color"],
            fill=True,
            fill_color=style["color"],
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{name} (PK {pk_abs:.3f})",
        ).add_to(layer)


# --- ADD TO MAP ---
for layer in LAYERS.values():
    layer.add_to(m)

# =============================================================================
# LAYER CONTROL  +  LEGENDA
# =============================================================================
folium.LayerControl(collapsed=False, position="topright").add_to(m)

legend_html = """
<div style="
    position:fixed; bottom:30px; left:30px; z-index:1000;
    background:white; border-radius:8px; padding:12px 16px;
    box-shadow:0 2px 10px rgba(0,0,0,0.25);
    font-family:sans-serif; font-size:13px; line-height:2;
">
  <b style="font-size:14px">Legenda infrastruttura</b><br>
  <span style="display:inline-block;width:30px;height:5px;
        background:#c0392b;border-radius:2px;vertical-align:middle"></span>
  &nbsp;Binario semplice<br>
  <span style="display:inline-block;width:30px;height:5px;
        background:#91f59b;border-radius:2px;vertical-align:middle"></span>
  &nbsp;Raddoppio nuovo (scenario)<br>
  <span style="display:inline-block;width:30px;height:5px;
        background:#08e01d;border-radius:2px;vertical-align:middle"></span>
  &nbsp;Raddoppio esistente (±100 m)<br>
  <hr style="margin:6px 0">
  <i style="font-size:11px;color:#666">
    Nel pannello in alto a destra:<br>
    attiva/disattiva uno scenario per volta.
  </i>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# =============================================================================
# SALVA MAPPA
# =============================================================================
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "val_venosta.html")
m.save(output_path)

print(f"\nMappa salvata — {len(points)} segnali, 3 scenari.\n{output_path}")