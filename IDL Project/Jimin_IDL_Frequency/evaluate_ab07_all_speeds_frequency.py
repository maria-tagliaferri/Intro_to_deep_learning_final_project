import argparse
from pathlib import Path

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
    sort_speed_key,
)


DEFAULT_OUTPUT_CSV = (
    Path("/home/metamobility2/JiminMM2/IDL Project/Jimin_IDL_Frequency")
    / "AB07_Leo_frequency_speed_metrics.csv"
)


def evaluate_model_for_speed(args, device, freq_label, spec, speed_dir, model):
    trial_dirs = sorted([p for p in speed_dir.iterdir() if p.is_dir() and p.name.startswith("trial_")])
    if args.trials:
        wanted_trials = set(args.trials)
        trial_dirs = [p for p in trial_dirs if p.name in wanted_trials]

    per_leg_truth = {"R": [], "L": []}
    config = hyperparam_config(spec["target_frequency"], spec["model_name"])
    start_index = config["window_size"] - 1

    print(f"Evaluating {freq_label} {speed_dir.name}: {len(trial_dirs)} trials")
    for trial_dir in trial_dirs:
        trial_data = load_trial_for_frequency(
            trial_dir,
            spec["model_dir"],
            spec["target_frequency"],
            label_cutoff_freq=args.label_cutoff_freq,
            zero_0mps_labels=args.zero_0mps_labels,
        )
        for leg in ["R", "L"]:
            input_tensor, label_raw = trial_data[leg]
            per_leg_truth[leg].append((input_tensor, label_raw))

    per_leg_metrics_truth = {"R": [], "L": []}
    per_leg_metrics_pred = {"R": [], "L": []}
    for leg in ["R", "L"]:
        input_tensor = torch.cat([item[0] for item in per_leg_truth[leg]], dim=0)
        label_raw = np.concatenate([item[1] for item in per_leg_truth[leg]])
        predictions = predict_series(
            model,
            input_tensor,
            spec["model_dir"],
            spec["target_frequency"],
            spec["model_name"],
            device,
            args.batch_size,
        )
        overlap = min(len(predictions), len(label_raw[start_index:]))
        per_leg_metrics_truth[leg].append(label_raw[start_index : start_index + overlap])
        per_leg_metrics_pred[leg].append(predictions[:overlap])

    rows = []
    for leg in ["R", "L"]:
        truth = np.concatenate(per_leg_metrics_truth[leg])
        pred = np.concatenate(per_leg_metrics_pred[leg])
        rmse, r2 = metric_pair(truth, pred)
        rows.append(
            {
                "subject": args.subject,
                "frequency": freq_label,
                "target_frequency_hz": spec["target_frequency"],
                "speed": speed_dir.name,
                "leg": leg,
                "trials": ";".join(p.name for p in trial_dirs),
                "n_samples": len(truth),
                "rmse": rmse,
                "r2": r2,
            }
        )

    both_truth = np.concatenate(per_leg_metrics_truth["R"] + per_leg_metrics_truth["L"])
    both_pred = np.concatenate(per_leg_metrics_pred["R"] + per_leg_metrics_pred["L"])
    rmse, r2 = metric_pair(both_truth, both_pred)
    rows.append(
        {
            "subject": args.subject,
            "frequency": freq_label,
            "target_frequency_hz": spec["target_frequency"],
            "speed": speed_dir.name,
            "leg": "both",
            "trials": ";".join(p.name for p in trial_dirs),
            "n_samples": len(both_truth),
            "rmse": rmse,
            "r2": r2,
        }
    )
    return rows


def evaluate(args):
    data_root = Path(args.data_root).expanduser()
    subject_dir = data_root / args.subject
    output_csv = Path(args.output_csv).expanduser()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

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

    rows = []
    speed_dirs = sorted([p for p in subject_dir.iterdir() if p.is_dir()], key=lambda p: sort_speed_key(p.name))
    for freq_label, spec in selected_specs.items():
        for speed_dir in speed_dirs:
            rows.extend(evaluate_model_for_speed(args, device, freq_label, spec, speed_dir, models[freq_label]))

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv, index=False)
    print(f"\nSaved frequency speed metrics to: {output_csv}")
    print(result_df.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate 50Hz, 25Hz, and 10Hz IDL frequency models on all AB07 trials grouped by speed."
    )
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--subject", default="AB07_Leo")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--frequencies", nargs="+", default=["50Hz", "25Hz", "10Hz"], choices=list(MODEL_SPECS))
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument("--label-cutoff-freq", type=float, default=4.0)
    parser.add_argument("--trials", nargs="*", default=None)
    parser.add_argument(
        "--zero-0mps-labels",
        action="store_true",
        help="Set 0mps labels to zero, matching the training dataloader behavior.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
