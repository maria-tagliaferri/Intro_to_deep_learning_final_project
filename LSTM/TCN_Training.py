# ICORR_TCN_Training.py
import torch
import os
import wandb

from TCN_Header_Model_LSTM import LSTMModel
from TCN_Header_Dataloader import DataHandler
from TCN_Header_Trainer import Trainer

wandb.login(key = 'a6db2cc1cfb2c7c9c8baaad4129c1398c5d4b8bd')
use_sweep = False  # Set to True to run sweep, False for single run
# Define the sweep configuration outside the main function
sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'RMSE',
        'goal': 'minimize'
    },
    'parameters': {
        # 'dataset_proportion': {'values': [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.2]},
        'window_size': {'values': [50, 100, 150, 200]},
        'hidden_size': {'values': [32, 64, 128]},
    }
}

# Base hyperparameters
hyperparam_config = {
    'wandb_project_name': 'Biotorque_trainsweep',
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

def train():

    # Initialize wandb run with hyperparameters
    wandb_run = wandb.init(config=hyperparam_config, project=hyperparam_config['wandb_project_name'], name=hyperparam_config['wandb_session_name'])

    # Access wandb.config after initializing wandb
    config = wandb.config

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    # Update hyperparameters with wandb config (useful if overridden)
    hyperparam_config.update(dict(config))

    # Create directory for results & plots
    save_dir = '/home/metamobility2/Maria_TCN/Biotorque AB06-AB09/LSTM/resultssweep'
    save_sub_dir = hyperparam_config['wandb_session_name']  
    save_dir = os.path.join(save_dir, save_sub_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Model Initialization
    model = LSTMModel(hyperparam_config).to(device)
    
    # Load pretrained model if transfer learning is enabled
    if hyperparam_config['transfer_learning']:
        pretrained_model_path = None
        model.load_state_dict(torch.load(os.path.join(pretrained_model_path, 'AB_model.pt'), map_location=device))
        print("\nPretrained model loaded: ", pretrained_model_path)
        # #Freeze the TCN part of the model
        # for param in model.tcn.parameters():
        #     param.requires_grad = False
    else:
        pretrained_model_path = None

    # Initialize DataHandler
    data_root = '/home/metamobility2/JiminMM2/Dataset/Biotorque_14Subjects/Synced_Biotorque_LG_Data'
    data_handler = DataHandler(data_root, hyperparam_config, pretrained_model_path)
    data_handler.load_data(
        train_data_partition=[
                        'AB01_Jimin',
                        'AB02_Rajiv',
                        'AB03_Amy',
                        'AB04_Changseob',
                        'AB05_Maria',
                        'AB06_Vaidehi',
                        'AB08_Adrian',
                        'AB09_Crystal',
                        'AB10_Pragya',
                        'AB11_Ryan',
                        'AB12_Ray',
                        'AB13_Hridayam',
                        'AB14_Evy'

        
        ],
        test_data_partition=[
                        'AB07_Leo',
        ],
    )
    data_handler.save_mean_std(save_dir)

    # Define Loss Function and Optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparam_config['init_lr'], weight_decay=1e-5)
    # Adjust optimizer to include only the FCNN parameters (freeze the TCN)
    # optimizer = torch.optim.Adam(model.linear.parameters(), lr=hyperparam_config['init_lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)

    # Initialize Trainer
    trainer = Trainer(device, model, wandb_run, criterion, optimizer, scheduler, data_handler, hyperparam_config, save_dir)

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    # Finish wandb wandb_run
    wandb_run.finish()

if __name__ == '__main__':
    
    if use_sweep:
        # Initialize the sweep
        sweep_id = wandb.sweep(sweep_config, project= hyperparam_config['wandb_project_name'])
        # Start the sweep agent
        wandb.agent(sweep_id, function=train)
    else:
        # For a single training wandb_run, call train() directly
        train()
