from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INFRASTRUCTURE = PROJECT_ROOT / "infra_data"
HEADWAYS = PROJECT_ROOT / "headways"

SCENARIO = "1a"

SCENARIOS = {
    "0": {"base", "existing"},
    "1a": {"base", "existing", "dt_tel_pla", "dt_natkomp_sta", "dt_sta_cia", "dt_cold_sblLac", "dt_lasa_oris"},
    "1b": {"base", "existing", "dt_tel_pla", "dt_sta_cia", "dt_cold_sblLac", "dt_lasa_oris"},
    "1c": {"base", "existing", "dt_me_lag", "dt_tel_pla", "dt_natkomp_sta", "dt_sta_cia", "dt_cold_sblLac", "dt_lasa_oris"},
    "2a": {"base", "existing"},
    "all": None,
}

def load_network_csv(filename, index_col):
    df = pd.read_csv(INFRASTRUCTURE / filename, sep=";", decimal=",", index_col=index_col)
    if SCENARIO == "all":
        return df[df["edge_type"] != "connecting"] if "edge_type" in df.columns else df
    active_scenario = SCENARIOS[SCENARIO]
    return df[df["scenario"].fillna("").apply(lambda s: bool(set(map(str.strip, s.split(","))) & active_scenario))]

def get_scenario():
    return SCENARIO

def load_headways():
    return np.load(HEADWAYS / f"all_min_headways{SCENARIO}.npy", allow_pickle=True).item()