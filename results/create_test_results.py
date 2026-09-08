from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infra_data.scenarios import SCENARIOS

RESULTS_DIR = PROJECT_ROOT / "results"

rng = np.random.default_rng(42)

for i, scenario in enumerate(SCENARIOS):

    nominal = 74.5 - 0.45 * i + rng.normal(0, 0.45)

    operational = 1.015 + 0.0018 * i + rng.normal(0, 0.0015)

    if scenario == "0":
        cost = 0.0
    elif scenario == "all":
        cost = 12.0
    else:
        cost = 2.0 + 0.65 * i + rng.normal(0, 0.25)

    data = {
        "nominal": float(max(nominal, 1.0)),
        "operational": float(max(operational, 0.01)),
        "cost": float(max(cost, 0.0)),
    }

    scenario_dir = RESULTS_DIR / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)

    output_path = scenario_dir / f"{scenario}_results.npy"

    np.save(output_path, data, allow_pickle=True)

    print(
        f"{scenario:>4} -> {output_path} | "
        f"nominal={data['nominal']:.3f}, "
        f"operational={data['operational']:.5f}, "
        f"cost={data['cost']:.2f}"
    )

print("\nDone. Synthetic results created for all scenarios.")