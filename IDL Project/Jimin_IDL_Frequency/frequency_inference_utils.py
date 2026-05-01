from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import signal

from TCN_Header_Model import TCNModel


DATA_ROOT = Path(
    "/home/metamobility2/JiminMM2/Dataset/Biotorque_14Subjects/Synced_Biotorque_LG_Data"
)
PROJECT_DIR = Path("/home/metamobility2/JiminMM2/IDL Project/Jimin_IDL_Frequency")
MODEL_SPECS = {
    "50Hz": {"target_frequency": 50, "model_name": "IDL_Freq_50Hz", "model_dir": PROJECT_DIR / "IDL_Freq_50Hz"},
    "25Hz": {"target_frequency": 25, "model_name": "IDL_Freq_25Hz", "model_dir": PROJECT_DIR / "IDL_Freq_25Hz"},
    "10Hz": {"target_frequency": 10, "model_name": "IDL_Freq_10Hz", "model_dir": PROJECT_DIR / "IDL_Freq_10Hz"},
}


def hyperparam_config(target_frequency, model_name):
    return {
        "wandb_project_name": "Biotorque_LGRARD",
        "wandb_session_name": model_name,
        "input_size": 6,
        "source_frequency": 100,
        "target_frequency": target_frequency,
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


def downsample_sequence(data, source_frequency, target_frequency):
    if target_frequency == source_frequency:
        return data
    if target_frequency <= 0 or source_frequency <= 0:
        raise ValueError("source_frequency and target_frequency must be positive")
    if target_frequency > source_frequency:
        raise ValueError(
            f"target_frequency ({target_frequency}) must be <= source_frequency ({source_frequency})"
        )

    ratio = Fraction(str(target_frequency / source_frequency)).limit_denominator(1000)
    if data.ndim == 1:
        return signal.resample_poly(data, ratio.numerator, ratio.denominator)
    channels = [
        signal.resample_poly(data[:, i], ratio.numerator, ratio.denominator)
        for i in range(data.shape[1])
    ]
    return np.stack(channels, axis=1)


def load_vicon_hip_moment_data(label_file, record_time_sec, label_cutoff_freq=4):
    output_df = pd.read_csv(
        label_file,
        delimiter=",",
        skiprows=lambda x: x in range(0, record_time_sec * 1000 + 10),
        encoding_errors="ignore",
    )
    output_df = output_df.fillna(0)
    output_buffer = output_df.values[:, [54, 6]] / 1000
    output_buffer[:, 0] *= -1
    return lowpass_filter(output_buffer, order=4, cutoff_freq=label_cutoff_freq, sampling_freq=100)


def find_single_file(directory, keyword):
    matches = sorted(p for p in directory.iterdir() if keyword.lower() in p.name.lower())
    if not matches:
        raise FileNotFoundError(f"No file containing '{keyword}' found in {directory}")
    return matches[0]


def load_frequency_model(model_dir, model_name, target_frequency, device, epoch=None):
    model_file = model_dir / (
        f"{model_name}_epoch_{epoch}.pt" if epoch is not None else f"{model_name}.pt"
    )
    model = TCNModel(hyperparam_config(target_frequency, model_name))
    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded {model_name}: {model_file}")
    return model


def load_trial_for_frequency(trial_dir, model_dir, target_frequency, label_cutoff_freq=4, zero_0mps_labels=False):
    input_dir = trial_dir / "Input"
    label_dir = trial_dir / "Label"
    imu_file = find_single_file(input_dir, "imu")
    label_file = sorted(label_dir.glob("*.csv"))[0]

    input_mean = np.load(model_dir / "input_mean.npy")
    input_std = np.load(model_dir / "input_std.npy")

    input_imu = pd.read_csv(imu_file)
    input_r = input_imu[
        [
            "Thigh_R_Acc_X",
            "Thigh_R_Acc_Y",
            "Thigh_R_Acc_Z",
            "Thigh_R_Gyr_X",
            "Thigh_R_Gyr_Y",
            "Thigh_R_Gyr_Z",
        ]
    ].values
    input_l = input_imu[
        [
            "Thigh_L_Acc_X",
            "Thigh_L_Acc_Y",
            "Thigh_L_Acc_Z",
            "Thigh_L_Gyr_X",
            "Thigh_L_Gyr_Y",
            "Thigh_L_Gyr_Z",
        ]
    ].values
    input_l[:, [1, 3, 5]] *= -1

    record_time_sec = int(input_r.shape[0] / 100)
    labels = load_vicon_hip_moment_data(label_file, record_time_sec, label_cutoff_freq)
    if zero_0mps_labels and trial_dir.parent.name == "0mps":
        labels[:] = 0

    input_r = downsample_sequence(input_r, 100, target_frequency)
    input_l = downsample_sequence(input_l, 100, target_frequency)
    label_r = downsample_sequence(labels[:, 0], 100, target_frequency)
    label_l = downsample_sequence(labels[:, 1], 100, target_frequency)

    input_r_tensor = torch.tensor((input_r - input_mean) / input_std, dtype=torch.float32)
    input_l_tensor = torch.tensor((input_l - input_mean) / input_std, dtype=torch.float32)

    return {
        "R": (input_r_tensor, label_r),
        "L": (input_l_tensor, label_l),
        "raw_labels_100hz": labels,
    }


def predict_series(model, input_tensor, model_dir, target_frequency, model_name, device, batch_size=1024):
    config = hyperparam_config(target_frequency, model_name)
    window_size = config["window_size"]
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
        return (0, float(speed[:-3].replace("p", ".")), speed)
    return (1, 0.0, speed)
