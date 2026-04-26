# Import necessary libraries
import torch
import numpy as np
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from TCN_Header_Model import TCNModel
from TCN_Header_Dataloader import LoadData  # Import LoadData directly

# def lowpass_filter(data, order=4, cutoff_freq=6, sampling_freq=100):
#     """
#     Apply a low-pass Butterworth filter to the data.
    
#     Parameters:
#         data (np.ndarray): Input data to be filtered.
#         order (int): Order of the filter.
#         cutoff_freq (float): Cutoff frequency in Hz.
#         sampling_freq (float): Sampling frequency in Hz.
        
#     Returns:
#         np.ndarray: Filtered data.
#     """
#     from scipy.signal import butter, filtfilt

#     nyquist = 0.5 * sampling_freq
#     normal_cutoff = cutoff_freq / nyquist
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     filtered_data = filtfilt(b, a, data)
#     return filtered_data

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    # Hyperparameter configuration
    hyperparam_config = {
            'input_size': 14, # 12 for IMU (right and pelvis), 2 for hip angle and velocity
            'output_size': 1, # 1 for right hip torque
            'architecture': 'TCN',
            
            'transfer_learning': False,
            'dataset_proportion': 1, # dataset proportion for training
            
            'epochs': 30,
            'batch_size': 32,
            'init_lr': 5e-4,
            'dropout': 0.15,
            'validation_split': 0.1,
            'window_size': 95,
            'number_of_layers': 2,
            'num_channels': [50, 50, 50, 50, 50],
            'kernel_size': 5,
            'dilations': [1, 2, 4, 8, 16],
            'number_of_workers': 10
        }

    # Directory paths
    save_dir = '/home/metamobility2/Maria_TCN/Biotorque AB06-AB09/training_result_all/unilateral'
    model_path = os.path.join(save_dir, 'unilateral.pt')

    # Load saved mean and std for normalization
    imu_mean = np.load(os.path.join(save_dir, 'input_mean.npy'))
    imu_std = np.load(os.path.join(save_dir, 'input_std.npy'))
    ik_mean = np.load(os.path.join(save_dir, 'label_mean.npy'))
    ik_std = np.load(os.path.join(save_dir, 'label_std.npy'))

    data_root =  '/home/metamobility2/JiminMM2/Dataset/Biotorque_14Subjects/Synced_Biotorque_LG_Data'

    # Load test data
    test_data_partition = ['AB07_Leo']  # Specify your test dataset partition
    test_data = LoadData(
        root=data_root,
        partitions=test_data_partition,
        window_size=hyperparam_config['window_size'],
        data_type="test_data",
        dataset_proportion=1,  
        input_mean=imu_mean,
        input_std=imu_std,
        label_mean=ik_mean,
        label_std=ik_std,
    )

    # Create Test DataLoader
    test_loader = DataLoader(
        dataset=test_data,
        num_workers=hyperparam_config['number_of_workers'],
        batch_size=hyperparam_config['batch_size'],
        pin_memory=True,
        shuffle=False
    )

    # Instantiate the TCN model
    model = TCNModel(hyperparam_config).to(device)

    # Load the trained model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Prepare mean and std tensors
    ik_mean_tensor = torch.tensor(ik_mean, device=device)
    ik_std_tensor = torch.tensor(ik_std, device=device)

    # Inference
    ik_pred_list = []
    ik_true_list = []
    with torch.no_grad():
        for imu, ik in tqdm(test_loader, desc='Inference'):
            imu = imu.to(device)
            ik = ik.to(device)
            logits = model(imu)

            # Denormalize predictions and targets
            preds_denorm = logits * ik_std_tensor + ik_mean_tensor
            ik_denorm = ik * ik_std_tensor + ik_mean_tensor

            ik_pred_list.append(preds_denorm.cpu().numpy())
            ik_true_list.append(ik_denorm.cpu().numpy())

    # # Concatenate predictions and true values
    ik_pred = np.concatenate(ik_pred_list, axis=0)
    ik_true = np.concatenate(ik_true_list, axis=0)

    # Extract data for the gait cycle

    # ik_pred = ik_pred[34:142]
    # ik_true = ik_true[34:142]
    ik_pred = ik_pred[2204:2286]
    ik_true = ik_true[2204:2286]

    # Save predictions and ground truth as CSV files
    predictions_csv_path = os.path.join(save_dir, 'prediction_estimation.csv')
    ground_truth_csv_path = os.path.join(save_dir, 'prediction_ground_truth.csv')

    # Save predictions
    np.savetxt(predictions_csv_path, ik_pred, delimiter=',', header='Predicted_Torque', comments='')
    np.savetxt(ground_truth_csv_path, ik_true, delimiter=',', header='True_Torque', comments='')

    print(f"Predictions saved to {predictions_csv_path}")
    print(f"Ground truth saved to {ground_truth_csv_path}")

    # Plot predictions vs ground truth
    plt.figure()
    plt.plot(ik_true, label='True Torque')
    plt.plot(ik_pred, label='Predicted Torque')
    plt.title('Predicted vs True Joint Torque')
    plt.xlabel('Sample')
    plt.ylabel('Torque (Nm)')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'torque_prediction.png'))
    plt.close()

    # Calculate RMSE
    rmse = np.sqrt(np.mean((ik_pred - ik_true) ** 2))
    ss_total = np.sum((ik_true - np.mean(ik_true)) ** 2)
    ss_residual = np.sum((ik_true - ik_pred) ** 2)
    r_squared = 1 - (ss_residual / ss_total)

    print(f"RMSE: {rmse:.4f}")
    print(f"R-squared: {r_squared:.4f}")

    print("Inference completed and results saved.")

if __name__ == '__main__':
    main()