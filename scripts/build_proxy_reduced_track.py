"""Build a formal proxy-reduced validation artifact from robustness results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "proposal" / "figures"


def main() -> None:
    robustness = json.loads((MODELS_DIR / "robustness.json").read_text())
    selected = robustness["proxy_core_removed"]
    full = robustness["full"]
    artifact = {
        "selected_track": "proxy_core_removed",
        "selected_track_metrics": selected,
        "full_track_metrics": full,
        "delta_vs_full": {
            "accuracy": round(full["accuracy"] - selected["accuracy"], 4),
            "macro_f1": round(full["macro_f1"] - selected["macro_f1"], 4),
            "weighted_f1": round(full["weighted_f1"] - selected["weighted_f1"], 4),
            "log_loss": round(selected["log_loss"] - full["log_loss"], 4),
        },
        "interpretation": (
            "This stricter track removes features closest to the current labeling rules. "
            "Its weaker score quantifies the remaining circularity gap."
        ),
    }
    (MODELS_DIR / "proxy_reduced_validation.json").write_text(
        json.dumps(artifact, indent=2),
        encoding="utf-8",
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Full", "Proxy-reduced"]
    values = [full["macro_f1"], selected["macro_f1"]]
    bars = ax.bar(labels, values, color=["#1f77b4", "#d62728"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Proxy-Reduced Validation Track")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.4f}", ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "proxy_reduced_validation.png", dpi=150)
    plt.close(fig)
    print("Proxy-reduced validation artifact written.")


if __name__ == "__main__":
    main()
