"""Plot training and validation losses from the saved local run logs."""
import argparse
import json
from pathlib import Path


def read_metrics(directory):
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No metrics in {directory}")
    if any(b["tokens"] <= a["tokens"] for a, b in zip(rows, rows[1:])):
        raise ValueError(f"Metrics are not in increasing training-token order: {directory}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("variant", type=Path)
    parser.add_argument("--out", type=Path, default=Path("comparison.png"))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), constrained_layout=True)
    for directory, label in ((args.baseline, "Baseline"), (args.variant, "Modification")):
        rows = read_metrics(directory)
        for axis, metric in zip(axes, ("train_loss", "val_loss", "val_bpb", "grad_norm")):
            values = [row for row in rows if row.get(metric) is not None]
            axis.plot([row["tokens"] / 1e6 for row in values], [row[metric] for row in values], label=label)
    for axis, title, ylabel in zip(
        axes,
        ("Training", "Validation", "Validation", "Gradient norm"),
        ("Loss (nats/token, logged batch)", "Loss (nats/token)", "Bits per byte", "Norm before clipping"),
    ):
        axis.set(title=title, xlabel="Training tokens (millions)", ylabel=ylabel)
        axis.grid(alpha=0.2)
        axis.legend()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(args.out)


if __name__ == "__main__":
    main()
