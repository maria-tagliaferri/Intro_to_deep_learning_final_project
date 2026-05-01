import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from frequency_inference_utils import (
    DATA_ROOT,
    MODEL_SPECS,
    hyperparam_config,
    load_frequency_model,
    load_trial_for_frequency,
    metric_pair,
    predict_series,
)


DEFAULT_OUTPUT_DIR = Path("/home/metamobility2/JiminMM2/IDL Project/Jimin_IDL_Frequency/trial_plots")
COLORS = {"50Hz": "tab:blue", "25Hz": "tab:orange", "10Hz": "tab:green"}


def evaluate_trial_for_model(args, device, trial_dir, freq_label, spec, model):
    trial_data = load_trial_for_frequency(
        trial_dir,
        spec["model_dir"],
        spec["target_frequency"],
        label_cutoff_freq=args.label_cutoff_freq,
        zero_0mps_labels=args.zero_0mps_labels,
    )
    config = hyperparam_config(spec["target_frequency"], spec["model_name"])
    start_index = config["window_size"] - 1
    result = {}
    result["gt_100hz"] = {
        "R": trial_data["raw_labels_100hz"][:, 0],
        "L": trial_data["raw_labels_100hz"][:, 1],
        "time": np.arange(len(trial_data["raw_labels_100hz"])) / 100,
    }

    for leg in args.legs:
        input_tensor, label_raw = trial_data[leg]
        try:
            predictions = predict_series(
                model,
                input_tensor,
                spec["model_dir"],
                spec["target_frequency"],
                spec["model_name"],
                device,
                args.batch_size,
            )
        except ValueError as exc:
            print(f"Skipping {freq_label} {leg}: {exc}")
            result[leg] = None
            continue
        overlap = min(len(predictions), len(label_raw[start_index:]))
        truth = label_raw[start_index : start_index + overlap]
        pred = predictions[:overlap]
        rmse, r2 = metric_pair(truth, pred)
        result[leg] = {
            "truth": truth,
            "pred": pred,
            "time": np.arange(start_index, start_index + overlap) / spec["target_frequency"],
            "rmse": rmse,
            "r2": r2,
        }
    return result


def save_metrics_csv(metrics_rows, output_dir, subject, speed, trial):
    metrics_csv = output_dir / f"{subject}_{speed}_{trial}_frequency_trial_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False)
    return metrics_csv


def plot_leg(args, output_dir, subject, speed, trial, leg, results_by_frequency):
    plt.figure(figsize=(15, 4))

    gt_source = next(iter(results_by_frequency.values()))["gt_100hz"]
    plt.plot(
        gt_source["time"],
        gt_source[leg],
        label="GT (100Hz, 4Hz LPF)",
        color="black",
        linestyle="--",
        linewidth=1.5,
    )

    for freq_label, result in results_by_frequency.items():
        series = result[leg]
        if series is None:
            continue
        label = f"{freq_label} pred (RMSE={series['rmse']:.3f}, R2={series['r2']:.3f})"
        plt.plot(series["time"], series["pred"], label=label, color=COLORS.get(freq_label))

    available_frequencies = [
        freq_label for freq_label, result in results_by_frequency.items() if result[leg] is not None
    ]
    if not available_frequencies:
        raise ValueError(f"No available predictions to plot for leg {leg}")

    if args.xlim is not None:
        plt.xlim(args.xlim[0], args.xlim[1])
    plt.xlabel("Time (s)")
    plt.ylabel("Hip Moment (N-m/kg)")
    plt.title(f"{subject} {speed} {trial} {leg} - Frequency Model Comparison")
    plt.legend()
    plt.tight_layout()

    plot_path = output_dir / f"{subject}_{speed}_{trial}_{leg}_frequency_comparison.png"
    plt.savefig(plot_path, dpi=300)
    if args.show:
        plt.show()
    plt.close()
    return plot_path


def run(args):
    data_root = Path(args.data_root).expanduser()
    trial_dir = data_root / args.subject / args.speed / args.trial
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_specs = {k: MODEL_SPECS[k] for k in args.frequencies}
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    models = {
        freq_label: load_frequency_model(
            spec["model_dir"],
            spec["model_name"],
            spec["target_frequency"],
            device,
            args.epoch,
        )
        for freq_label, spec in selected_specs.items()
    }

    results_by_frequency = {
        freq_label: evaluate_trial_for_model(args, device, trial_dir, freq_label, spec, models[freq_label])
        for freq_label, spec in selected_specs.items()
    }

    metrics_rows = []
    for freq_label, leg_results in results_by_frequency.items():
        for leg, series in leg_results.items():
            if leg == "gt_100hz":
                continue
            if series is None:
                metrics_rows.append(
                    {
                        "subject": args.subject,
                        "speed": args.speed,
                        "trial": args.trial,
                        "frequency": freq_label,
                        "target_frequency_hz": MODEL_SPECS[freq_label]["target_frequency"],
                        "leg": leg,
                        "n_samples": 0,
                        "rmse": float("nan"),
                        "r2": float("nan"),
                    }
                )
                continue
            metrics_rows.append(
                {
                    "subject": args.subject,
                    "speed": args.speed,
                    "trial": args.trial,
                    "frequency": freq_label,
                    "target_frequency_hz": MODEL_SPECS[freq_label]["target_frequency"],
                    "leg": leg,
                    "n_samples": len(series["truth"]),
                    "rmse": series["rmse"],
                    "r2": series["r2"],
                }
            )

    metrics_csv = save_metrics_csv(metrics_rows, output_dir, args.subject, args.speed, args.trial)
    plot_paths = [
        plot_leg(args, output_dir, args.subject, args.speed, args.trial, leg, results_by_frequency)
        for leg in args.legs
    ]

    print(f"Saved trial metrics to: {metrics_csv}")
    for path in plot_paths:
        print(f"Saved plot to: {path}")
    print(pd.DataFrame(metrics_rows).to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot 50Hz, 25Hz, and 10Hz frequency model predictions together for one AB07 trial."
    )
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--subject", default="AB07_Leo")
    parser.add_argument("--speed", default="1p2mps")
    parser.add_argument("--trial", default="trial_3")
    parser.add_argument("--legs", nargs="+", default=["R", "L"], choices=["R", "L"])
    parser.add_argument("--frequencies", nargs="+", default=["50Hz", "25Hz", "10Hz"], choices=list(MODEL_SPECS))
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument("--label-cutoff-freq", type=float, default=4.0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--xlim", nargs=2, type=float, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--zero-0mps-labels",
        action="store_true",
        help="Set 0mps labels to zero, matching the training dataloader behavior.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
