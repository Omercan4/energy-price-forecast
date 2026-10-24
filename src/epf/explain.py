"""What drives the forecast? SHAP values of the median model, as a bar chart and a JSON file."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shap  # noqa: E402

from epf import config  # noqa: E402
from epf.data import build  # noqa: E402
from epf.features import training_table  # noqa: E402
from epf.forecast import load_model  # noqa: E402


def explain(sample_days: int = 120) -> dict:
    """Mean absolute SHAP value per feature on the most recent days (in EUR/MWh)."""
    model = load_model()
    table = training_table(build.load_dataset())
    sample = table[table.index > table.index.max() - np.timedelta64(sample_days, "D")]
    explainer = shap.TreeExplainer(model.models[0.5])
    values = explainer.shap_values(sample[model.features])
    importance = dict(sorted(zip(model.features, np.abs(values).mean(axis=0).round(2)), key=lambda kv: -kv[1]))
    config.DOCS_DIR.mkdir(exist_ok=True)
    (config.DOCS_DIR / "feature_importance.json").write_text(json.dumps(importance, indent=1, default=float))

    top = list(importance.items())[:10][::-1]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh([name for name, _ in top], [float(v) for _, v in top], color="#1f6feb")
    ax.set_xlabel("mean |SHAP|, EUR/MWh (what moves the median forecast)")
    ax.set_title(f"Main drivers of the forecast (last {sample_days} days)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    (config.DOCS_DIR / "img").mkdir(exist_ok=True)
    fig.savefig(config.DOCS_DIR / "img" / "shap.png", dpi=150)
    plt.close(fig)
    return importance


if __name__ == "__main__":
    print(json.dumps(explain(), indent=1, default=float))
