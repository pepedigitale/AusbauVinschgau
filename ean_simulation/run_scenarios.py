from pathlib import Path
import os
import papermill as pm

BASE_DIR = Path(__file__).resolve().parent
NOTEBOOK = BASE_DIR / "ean_simulation.ipynb"
OUTPUT_DIR = BASE_DIR / "executed"


SCENARIOS = ["1a", "1b", "1c", "3a", "3b", "3c", "3d", "3e", "3ee", "3f", "4"]


OUTPUT_DIR.mkdir(exist_ok=True)

for scenario in SCENARIOS:
    scenario
    os.environ["SCENARIO"] = scenario

    output_notebook = OUTPUT_DIR / f"analysis_{scenario}.ipynb"

    pm.execute_notebook(
    str(NOTEBOOK),
    str(OUTPUT_DIR / f"ean_simulation_{scenario}.ipynb"),
    cwd=str(BASE_DIR),
)