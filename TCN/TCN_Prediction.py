
# Import necessary libraries
import torch
import numpy as np
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from TCN_Header_Model import TCNModel
from TCN_Header_Dataloader import LoadData  # Import LoadData directly

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    # Hyperparameter configuration
    hyperparam_config = {
        'input_size': 14,  # Number of sensor inputs
        'output_size': 1,  # Number of joint angles
        'window_size': 25,
        'batch_size': 16,
        'num_channels': [50, 50, 50, 50, 50],  # Channels for each TCN layer
        'kernel_size': 5,  # Kernel size for convolutional layers      
        'number_of_layers': 2,  # Number of TCN layers
        'dilations': [1, 2, 4, 8, 16],
        'number_of_workers': os.cpu_count(),
        'dropout': 0.2
    }

    # Directory paths
    save_dir = '/home/metamobility2/Maria_TCN/Biotorque AB06-AB09/training_result/unilateral'
    model_path = os.path.join(save_dir, 'unilateral.pt')

    # Load saved mean and std for normalization
    imu_mean = np.load(os.path.join(save_dir, 'input_mean.npy'))
    imu_std = np.load(os.path.join(save_dir, 'input_std.npy'))
    ik_mean = np.load(os.path.join(save_dir, 'label_mean.npy'))
    ik_std = np.load(os.path.join(save_dir, 'label_std.npy'))

    data_root = '/home/metamobility2/Maria_TCN/Biotorque AB06-AB09/Data'
    # Load test data
    test_data_partition = [
                        'AB09_Crystal'
                        ]  # Specify your test dataset partition
    test_data = LoadData(
        root=data_root,
        partitions=test_data_partition,
        window_size=hyperparam_config['window_size'],
        data_type="test_data",
        dataset_proportion=0.1,
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

    # Concatenate predictions and true values
    ik_pred = np.concatenate(ik_pred_list, axis=0)
    ik_true = np.concatenate(ik_true_list, axis=0)

    # Save predictions and ground truth as CSV files
    predictions_csv_path = os.path.join(save_dir, 'prediction_estimation.csv')
    ground_truth_csv_path = os.path.join(save_dir, 'prediction_ground_truth.csv')

    # Define headers for CSV files
    headers = ['Joint_0_Predicted', 'Joint_1_Predicted', 'Joint_2_Predicted', 'Joint_3_Predicted']
    ground_truth_headers = ['Joint_0_True', 'Joint_1_True', 'Joint_2_True', 'Joint_3_True']

    # Save predictions
    np.savetxt(predictions_csv_path, ik_pred, delimiter=',', header=','.join(headers), comments='')

    # Save ground truth
    np.savetxt(ground_truth_csv_path, ik_true, delimiter=',', header=','.join(ground_truth_headers), comments='')

    print(f"Predictions saved to {predictions_csv_path}")
    print(f"Ground truth saved to {ground_truth_csv_path}")

    # Plot predictions vs ground truth
    for i in range(hyperparam_config['output_size']):
        plt.figure()
        plt.plot(ik_true[:, i], label='True')
        plt.plot(ik_pred[:, i], label='Predicted')
        plt.title(f'Joint Angle {i}')
        plt.xlabel('Sample')
        plt.ylabel('Angle (degrees)')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'joint_{i}_prediction.png'))
        plt.close()

    # Calculate RMSE
    rmse = np.sqrt(np.mean((ik_pred - ik_true) ** 2, axis=0))
    overall_rmse = np.mean(rmse) 

    # Print RMSE for each joint
    for i in range(hyperparam_config['output_size']):
        print(f"RMSE for Joint {i}: {rmse[i]:.4f} degrees")

    # Print overall RMSE
    print(f"Overall RMSE: {overall_rmse:.4f} degrees")

    print("Inference completed and results saved.")

if __name__ == '__main__':
    main()