import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import signal

from TCN_Header_Model import TCNModel


DEFAULT_DATA_ROOT = Path(
    "/home/metamobility2/JiminMM2/Dataset/Biotorque_14Subjects/Synced_Biotorque_LG_Data"
)
DEFAULT_MODEL_DIR = Path(
    "/home/metamobility2/JiminMM2/IDL Project/Jimin_IDL_GyroOnly/IDL_GyroOnly"
)
DEFAULT_OUTPUT_CSV = DEFAULT_MODEL_DIR / "AB07_Leo_gyro_only_speed_metrics.csv"

HYPERPARAM_CONFIG = {
    "wandb_project_name": "Biotorque_LGRARD",
    "wandb_session_name": "IDL_GyroOnly",
    "input_size": 3,
    "output_size": 1,
    "architecture": "TCN",
    "transfer_learning": False,
    "dataset_proportion": 1.0,
    "resume_training": True,
    "epochs": 30,
    "batch_size": 32,
    "init_lr": 5e-6,
    "dropout": 0.15,
    "validation_split": 0.1,
    "window_size": 95,
    "number_of_layers": 2,
    "num_channels": [80, 80, 80, 80, 80],
    "kernel_size": 5,
    "dilations": [1, 2, 4, 8, 16],
    "number_of_workers": 10,
}


def lowpass_filter(data, order=4, cutoff_freq=4, sampling_freq=100):
    nyquist_freq = sampling_freq / 2
    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    b, a = signal.butter(order, normalized_cutoff_freq, btype="low")
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
    return filtered_data


def load_vicon_hip_moment_data(label_file, record_time_sec, cutoff_freq):
    output_df = pd.read_csv(
        label_file,
        delimiter=",",
        skiprows=lambda x: x in range(0, record_time_sec * 1000 + 10),
        encoding_errors="ignore",
    )
    output_df = output_df.fillna(0)
    output_buffer = output_df.values[:, [54, 6]] / 1000
    output_buffer[:, 0] *= -1
    return lowpass_filter(output_buffer, order=4, cutoff_freq=cutoff_freq, sampling_freq=100)


def load_model(model_dir, model_name, epoch, device):
    model_file = model_dir / (
        f"{model_name}_epoch_{epoch}.pt" if epoch is not None else f"{model_name}.pt"
    )
    model = TCNModel(HYPERPARAM_CONFIG)
    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded model: {model_file}")
    return model


def find_single_file(directory, keyword):
    matches = sorted(p for p in directory.iterdir() if keyword.lower() in p.name.lower())
    if not matches:
        raise FileNotFoundError(f"No file containing '{keyword}' found in {directory}")
    return matches[0]


def load_trial(trial_dir, model_dir, label_cutoff_freq, zero_0mps_labels):
    input_dir = trial_dir / "Input"
    label_dir = trial_dir / "Label"
    imu_file = find_single_file(input_dir, "imu")
    label_file = sorted(label_dir.glob("*.csv"))[0]

    input_mean = np.load(model_dir / "input_mean.npy")
    input_std = np.load(model_dir / "input_std.npy")

    input_imu = pd.read_csv(imu_file)
    input_r = input_imu[["Thigh_R_Gyr_X", "Thigh_R_Gyr_Y", "Thigh_R_Gyr_Z"]].values
    input_l = input_imu[["Thigh_L_Gyr_X", "Thigh_L_Gyr_Y", "Thigh_L_Gyr_Z"]].values
    input_l[:, [0, 2]] *= -1

    record_time_sec = int(input_r.shape[0] / 100)
    labels = load_vicon_hip_moment_data(label_file, record_time_sec, label_cutoff_freq)
    if zero_0mps_labels and trial_dir.parent.name == "0mps":
        labels[:] = 0

    input_r_tensor = torch.tensor((input_r - input_mean) / input_std, dtype=torch.float32)
    input_l_tensor = torch.tensor((input_l - input_mean) / input_std, dtype=torch.float32)

    return {
        "R": (input_r_tensor, labels[:, 0]),
        "L": (input_l_tensor, labels[:, 1]),
    }


def predict_series(model, input_tensor, model_dir, device, batch_size):
    window_size = HYPERPARAM_CONFIG["window_size"]
    total_length = input_tensor.shape[0]
    valid_count = total_length - window_size + 1
    if valid_count <= 0:
        raise ValueError(f"Trial is shorter than window_size={window_size}: {total_length}")

    label_mean = np.load(model_dir / "label_mean.npy").item()
    label_std = np.load(model_dir / "label_std.npy").item()
    predictions = np.zeros(valid_count)

    with torch.no_grad():
        for start in range(0, valid_count, batch_size):
            end = min(start + batch_size, valid_count)
            windows = torch.stack(
                [input_tensor[i : i + window_size].T for i in range(start, end)]
            ).to(device)
            pred_norm = model(windows).detach().cpu().numpy().reshape(-1)
            predictions[start:end] = pred_norm * label_std + label_mean

    return predictions


def metric_pair(ground_truth, predictions):
    rmse = float(np.sqrt(np.mean((ground_truth - predictions) ** 2)))
    ss_res = float(np.sum((ground_truth - predictions) ** 2))
    ss_tot = float(np.sum((ground_truth - np.mean(ground_truth)) ** 2))
    r2 = float("nan") if ss_tot == 0 else 1 - (ss_res / ss_tot)
    return rmse, r2


def sort_speed_key(speed):
    if speed == "0mps":
        return (0, 0.0, speed)
    if speed.endswith("mps"):
        numeric_speed = speed[:-3].replace("p", ".")
        return (0, float(numeric_speed), speed)
    return (1, 0.0, speed)


def evaluate(args):
    data_root = Path(args.data_root).expanduser()
    model_dir = Path(args.model_dir).expanduser()
    subject_dir = data_root / args.subject
    output_csv = Path(args.output_csv).expanduser()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(model_dir, args.model_name, args.epoch, device)

    rows = []
    speed_dirs = sorted([p for p in subject_dir.iterdir() if p.is_dir()], key=lambda p: sort_speed_key(p.name))
    for speed_dir in speed_dirs:
        trial_dirs = sorted([p for p in speed_dir.iterdir() if p.is_dir() and p.name.startswith("trial_")])
        if args.trials:
            wanted_trials = set(args.trials)
            trial_dirs = [p for p in trial_dirs if p.name in wanted_trials]

        per_leg_truth = {"R": [], "L": []}
        per_leg_pred = {"R": [], "L": []}

        print(f"Evaluating {speed_dir.name}: {len(trial_dirs)} trials")
        for trial_dir in trial_dirs:
            trial_data = load_trial(
                trial_dir,
                model_dir,
                label_cutoff_freq=args.label_cutoff_freq,
                zero_0mps_labels=args.zero_0mps_labels,
            )
            for leg, (input_tensor, label_raw) in trial_data.items():
                predictions = predict_series(model, input_tensor, model_dir, device, args.batch_size)
                start = HYPERPARAM_CONFIG["window_size"] - 1
                overlap = min(len(predictions), len(label_raw[start:]))
                per_leg_truth[leg].append(label_raw[start : start + overlap])
                per_leg_pred[leg].append(predictions[:overlap])

        for leg in ["R", "L"]:
            truth = np.concatenate(per_leg_truth[leg])
            pred = np.concatenate(per_leg_pred[leg])
            rmse, r2 = metric_pair(truth, pred)
            rows.append(
                {
                    "subject": args.subject,
                    "speed": speed_dir.name,
                    "leg": leg,
                    "trials": ";".join(p.name for p in trial_dirs),
                    "n_samples": len(truth),
                    "rmse": rmse,
                    "r2": r2,
                }
            )

        both_truth = np.concatenate(per_leg_truth["R"] + per_leg_truth["L"])
        both_pred = np.concatenate(per_leg_pred["R"] + per_leg_pred["L"])
        rmse, r2 = metric_pair(both_truth, both_pred)
        rows.append(
            {
                "subject": args.subject,
                "speed": speed_dir.name,
                "leg": "both",
                "trials": ";".join(p.name for p in trial_dirs),
                "n_samples": len(both_truth),
                "rmse": rmse,
                "r2": r2,
            }
        )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv, index=False)
    print(f"\nSaved speed metrics to: {output_csv}")
    print(result_df.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate IDL GyroOnly offline inference on all AB07 trials grouped by speed."
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--subject", default="AB07_Leo")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--model-name", default="IDL_GyroOnly")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
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
