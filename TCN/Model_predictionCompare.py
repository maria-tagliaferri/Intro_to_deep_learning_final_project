# Import necessary libraries
import torch
import numpy as np
from tqdm.auto import tqdm
import os
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy import signal


from TCN_Header_Model_LSTM import LSTMModel
from TCN_Header_DataloaderLSTM import LoadData as LoadDataLSTM  # Import LoadData directly

from TCN_Header_Model import TCNModel
from TCN_Header_Dataloader import LoadData  # Import LoadData directly
def lowpass_filter(data, order=4, cutoff_freq=6, sampling_freq=100):
        """
        Parameters:
        data: numpy array of shape (n_samples, n_features)
        order: filter order (default=4)
        cutoff_freq: cutoff frequency in Hz (default=6)
        sampling_freq: sampling frequency in Hz (default=100)
        """
        nyquist_freq = sampling_freq / 2
        normalized_cutoff_freq = cutoff_freq / nyquist_freq
        
        # Create the filter coefficients
        b, a = signal.butter(order, normalized_cutoff_freq, btype='low')
        
        # Initialize filtered data array
        filtered_data = np.zeros_like(data)
        filtered_data[:, 0] = data[:, 0]  # Keep the time column as is
        
        # Apply filter to each column
        for i in range(0, data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
        
        return filtered_data

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    # Hyperparameter configuration
    hyperparam_config_lstm = {
        'wandb_project_name': 'Biotorque_train',
        'wandb_session_name': 'unilateral',
        'input_size': 14, # 12 for IMU (right and pelvis), 2 for hip angle and velocity, 8 if exlcusing pelvis 14 if not 
        'output_size': 1, # 1 for right hip torque
        'architecture': 'LSTM',
        
        'transfer_learning': False,
        'dataset_proportion': 1, # dataset proportion for training
        
        'epochs': 30,
        'batch_size': 32,
        'init_lr': 5e-4,
        'dropout': 0.2,
        'validation_split': 0.1,
        'window_size': 95,
        'number_of_layers': 2,
        'hidden_size': 50,
        'number_of_workers': 10
    }

    hyperparam_config_tcn = {
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
    save_dir_lstm = '/home/metamobility2/Maria_TCN/Biotorque AB06-AB09/LSTM/results/unilateral'
    model_path_lstm = os.path.join(save_dir_lstm, 'LSTM.pt')

    save_dir_tcn = '/home/metamobility2/Maria_TCN/Biotorque AB06-AB09/training_result_all/unilateral'
    model_path_tcn = os.path.join(save_dir_tcn, 'unilateral.pt')

    # Load saved mean and std for normalization
    imu_mean_lstm = np.load(os.path.join(save_dir_lstm, 'input_mean.npy'))
    imu_std_lstm = np.load(os.path.join(save_dir_lstm, 'input_std.npy'))
    ik_mean_lstm = np.load(os.path.join(save_dir_lstm, 'label_mean.npy'))
    ik_std_lstm = np.load(os.path.join(save_dir_lstm, 'label_std.npy'))

    imu_mean_tcn = np.load(os.path.join(save_dir_tcn, 'input_mean.npy'))
    imu_std_tcn = np.load(os.path.join(save_dir_tcn, 'input_std.npy'))
    ik_mean_tcn = np.load(os.path.join(save_dir_tcn, 'label_mean.npy'))
    ik_std_tcn = np.load(os.path.join(save_dir_tcn, 'label_std.npy'))

    data_root =  '/home/metamobility2/JiminMM2/Dataset/Biotorque_14Subjects/Synced_Biotorque_LG_Data'

    # Load test data
    test_data_partition = ['AB07_Leo']  # Specify your test dataset partition
    test_data_lstm = LoadDataLSTM(
        root=data_root,
        partitions=test_data_partition,
        window_size=hyperparam_config_lstm['window_size'],
        data_type="test_data",
        dataset_proportion=1,  
        input_mean=imu_mean_lstm,
        input_std=imu_std_lstm,
        label_mean=ik_mean_lstm,
        label_std=ik_std_lstm,
    )

    test_data_tcn = LoadData(
        root=data_root,
        partitions=test_data_partition,
        window_size=hyperparam_config_tcn['window_size'],
        data_type="test_data",
        dataset_proportion=1,  
        input_mean=imu_mean_tcn,
        input_std=imu_std_tcn,
        label_mean=ik_mean_tcn,
        label_std=ik_std_tcn,
    )

    # Create Test DataLoader
    test_loader_lstm = DataLoader(
        dataset=test_data_lstm,
        num_workers=hyperparam_config_lstm['number_of_workers'],
        batch_size=hyperparam_config_lstm['batch_size'],
        pin_memory=True,
        shuffle=False
    )

    test_loader_tcn = DataLoader(
        dataset=test_data_tcn,
        num_workers=hyperparam_config_tcn['number_of_workers'],
        batch_size=hyperparam_config_tcn['batch_size'],
        pin_memory=True,
        shuffle=False
    )

    # Instantiate the LSTM model
    model_lstm = LSTMModel(hyperparam_config_lstm).to(device)
    model_tcn = TCNModel(hyperparam_config_tcn).to(device)

    # Load the trained model
    model_lstm.load_state_dict(torch.load(model_path_lstm, map_location=device))
    model_lstm.eval()

    model_tcn.load_state_dict(torch.load(model_path_tcn, map_location=device))
    model_tcn.eval()

    # Prepare mean and std tensors
    ik_mean_tensor_lstm = torch.tensor(ik_mean_lstm, device=device)
    ik_std_tensor_lstm = torch.tensor(ik_std_lstm, device=device)

    ik_mean_tensor_tcn = torch.tensor(ik_mean_tcn, device=device)
    ik_std_tensor_tcn = torch.tensor(ik_std_tcn, device=device)


    # Inference
    ik_pred_list_lstm = []
    ik_true_list_lstm = []
    with torch.no_grad():
        for imu, ik in tqdm(test_loader_lstm, desc='Inference'):
            imu = imu.to(device)
            ik = ik.to(device)
            logits_lstm = model_lstm(imu)

            # Denormalize predictions and targets
            preds_denorm_lstm = logits_lstm * ik_std_tensor_lstm + ik_mean_tensor_lstm
            ik_denorm_lstm = ik * ik_std_tensor_lstm + ik_mean_tensor_lstm

            ik_pred_list_lstm.append(preds_denorm_lstm.cpu().numpy())
            ik_true_list_lstm.append(ik_denorm_lstm.cpu().numpy())


    ik_pred_list_tcn = []
    ik_true_list_tcn = []
    with torch.no_grad():
        for imu, ik in tqdm(test_loader_tcn, desc='Inference'):
            imu = imu.to(device)
            ik = ik.to(device)
            logits_tcn = model_tcn(imu)

            # Denormalize predictions and targets
            preds_denorm_tcn = logits_tcn * ik_std_tensor_tcn + ik_mean_tensor_tcn
            ik_denorm_tcn = ik * ik_std_tensor_tcn + ik_mean_tensor_tcn

            ik_pred_list_tcn.append(preds_denorm_tcn.cpu().numpy())
            ik_true_list_tcn.append(ik_denorm_tcn.cpu().numpy())
    # # Concatenate predictions and true values
    ik_pred_lstm = np.concatenate(ik_pred_list_lstm, axis=0)
    ik_true_lstm = np.concatenate(ik_true_list_lstm, axis=0)

    ik_pred_tcn = np.concatenate(ik_pred_list_tcn, axis=0)
    ik_true_tcn = np.concatenate(ik_true_list_tcn, axis=0)
    # Extract data for the gait cycle

    # ik_pred = ik_pred[34:142]
    # ik_true = ik_true[34:142]
    ik_pred_lstm = lowpass_filter(ik_pred_lstm[1:1500])
    ik_true_lstm = ik_true_lstm[1:1500]
    ik_pred_tcn = lowpass_filter(ik_pred_tcn[1:1500])
    ik_true_tcn = ik_true_tcn[1:1500]

    save_dir = '/home/metamobility2/Maria_TCN/Biotorque AB06-AB09/project/comparison_result'
    # Save predictions and ground truth as CSV files
    predictions_csv_path = os.path.join(save_dir, 'prediction_estimation.csv')
    ground_truth_csv_path = os.path.join(save_dir, 'prediction_ground_truth.csv')

    # Save predictions and ground truth in a CSV layout similar to Model_prediction.py.
    predictions_to_save = np.column_stack(
        [ik_pred_lstm.reshape(-1), ik_pred_tcn.reshape(-1)]
    )
    ground_truth_to_save = ik_true_lstm.reshape(-1, 1)

    np.savetxt(
        predictions_csv_path,
        predictions_to_save,
        delimiter=',',
        header='Predicted_Torque_LSTM,Predicted_Torque_TCN',
        comments=''
    )
    np.savetxt(
        ground_truth_csv_path,
        ground_truth_to_save,
        delimiter=',',
        header='True_Torque',
        comments=''
    )

    print(f"Predictions saved to {predictions_csv_path}")
    print(f"Ground truth saved to {ground_truth_csv_path}")

    # Plot predictions vs ground truth
    plt.figure()
    plt.plot(ik_true_lstm, label='True Torque')
    plt.plot(ik_pred_lstm, label='Predicted Torque LSTM')
    plt.plot(ik_pred_tcn, label='Predicted Torque TCN')
    plt.title('Predicted vs True Joint Torque')
    plt.xlabel('Time ')
    plt.ylabel('Torque (Nm)')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'torque_prediction.svg'))
    plt.close()

    # Calculate RMSE
    rmse_lstm = np.sqrt(np.mean((ik_pred_lstm - ik_true_lstm) ** 2))
    ss_total_lstm = np.sum((ik_true_lstm - np.mean(ik_true_lstm)) ** 2)
    ss_residual_lstm = np.sum((ik_true_lstm - ik_pred_lstm) ** 2)
    r_squared_lstm = 1 - (ss_residual_lstm / ss_total_lstm)

    rmse_tcn = np.sqrt(np.mean((ik_pred_tcn - ik_true_tcn) ** 2))
    ss_total_tcn = np.sum((ik_true_tcn - np.mean(ik_true_tcn)) ** 2)
    ss_residual_tcn = np.sum((ik_true_tcn - ik_pred_tcn) ** 2)
    r_squared_tcn = 1 - (ss_residual_tcn / ss_total_tcn)

    print(f"RMSE LSTM: {rmse_lstm:.4f}")
    print(f"R-squared LSTM: {r_squared_lstm:.4f}")
    print(f"RMSE TCN: {rmse_tcn:.4f}")
    print(f"R-squared TCN: {r_squared_tcn:.4f}")

    print("Inference completed and results saved.")

if __name__ == '__main__':
    main()