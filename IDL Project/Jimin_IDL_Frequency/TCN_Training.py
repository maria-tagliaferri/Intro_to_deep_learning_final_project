# ICORR_TCN_Training.py
import torch
import os
import wandb

from TCN_Header_Model import TCNModel, LSTMModel
from TCN_Header_Dataloader import DataHandler
from TCN_Header_Trainer import Trainer


model_training_name = "IDL_Freq_10Hz"  # Name for the training session

source_frequency = 100
target_frequency = 10

training_data_root = '/home/metamobility2/JiminMM2/Dataset/Biotorque_14Subjects/Synced_Biotorque_LG_Data'

# model_save_dir = '/home/metamobility2/JiminMM2/'
model_save_dir = '/home/metamobility2/JiminMM2/Jimin_IDL_Frequency'

use_sweep = False  # Set to True to run sweep, False for single run
# Define the sweep configuration outside the main function
sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'RMSE',
        'goal': 'minimize'
    },
    'parameters': {
        'num_channels': {'values': [[50, 50, 50], [50, 50, 50, 50], [50, 50, 50, 50, 50],
                                    [30, 30, 30], [30, 30, 30, 30], [30, 30, 30, 30, 30],
                                    [70, 70, 70], [70, 70, 70, 70], [70, 70, 70, 70, 70]]},
        'kernel_size': {'values': [5, 7, 10, 13]},
    }
}

# Base hyperparameters
hyperparam_config = {
    'wandb_project_name': 'Biotorque_LGRARD',
    'wandb_session_name': model_training_name,
    'input_size': 6,
    'source_frequency': source_frequency,
    'target_frequency': target_frequency,
    'output_size': 1, # 1 for right hip torque
    'architecture': 'TCN',
    
    'transfer_learning': False,
    'dataset_proportion': 1.0, # trial data proportion for training (1 for using all, 0.5 for half, etc.)
    'resume_training': True,  # Set to True to resume from checkpoint if available
    
    'epochs': 30,
    'batch_size': 32,
    'init_lr': 5e-6,
    'dropout': 0.15,
    'validation_split': 0.1,
    'window_size': 95,
    'number_of_layers': 2,
    'num_channels': [80, 80, 80, 80, 80],
    'kernel_size': 5,
    'dilations': [1, 2, 4, 8, 16],
    'number_of_workers': 10
}

def train():
    # Create directory for results & plots first
    save_dir = model_save_dir
    save_sub_dir = hyperparam_config['wandb_session_name']
    save_dir = os.path.join(save_dir, save_sub_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Check for existing checkpoint
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pt')
    resume_from_checkpoint = hyperparam_config['resume_training'] and os.path.exists(checkpoint_path)
    
    # Generate or load wandb run ID
    wandb_run_id = None
    if resume_from_checkpoint:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            wandb_run_id = checkpoint.get('wandb_run_id', None)
            print(f"Found checkpoint at epoch {checkpoint['epoch']+1}")

        except Exception as e:
            print(f"Failed to load checkpoint metadata: {e}")
            print("Starting fresh...")
            resume_from_checkpoint = False
    
    if not wandb_run_id:
        import uuid
        wandb_run_id = str(uuid.uuid4())[:8]

    # Initialize wandb run with hyperparameters
    wandb_run = wandb.init(
        config=hyperparam_config, 
        project=hyperparam_config['wandb_project_name'], 
        name=hyperparam_config['wandb_session_name'],
        resume='allow',
        id=wandb_run_id
    )

    # Access wandb.config after initializing wandb
    config = wandb.config

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    # Update hyperparameters with wandb config (useful if overridden)
    hyperparam_config.update(dict(config))

    # Model Initialization
    model = TCNModel(hyperparam_config).to(device)
    
    # Load pretrained model if transfer learning is enabled
    if hyperparam_config['transfer_learning']:
        pretrained_model_path = '/home/metamobility/Changseob/biotorque/training_result/baseline_TCN'
        model.load_state_dict(torch.load(os.path.join(pretrained_model_path, 'baseline_TCN.pt'), map_location=device))
        print("\nPretrained model loaded: ", pretrained_model_path)
        #Freeze the TCN part of the model
        for param in model.tcn.parameters():
            param.requires_grad = False
    else:
        pretrained_model_path = None

    # Initialize DataHandler
    # data_root = '/home/metamobility/Changseob/biotorque/Synced_Biotorque_LG_Data'
    # data_root = '/home/metamobility2/JiminMM2/Synced_LGRARD'
    
    data_root = training_data_root
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
        speed_condition=[
            '0mps', '0p2mps', '0p4mps', '0p6mps', '0p8mps', '1p0mps', '1p2mps', '1p4mps', 'transient_15sec', 'transient_30sec',
        ],
        task_condition=[
            '0mps', '0p2mps', '0p4mps', '0p6mps', '0p8mps', '1p0mps', '1p2mps', '1p4mps', 'transient_15sec', 'transient_30sec',
        ],
        test_data_partition=[
            "AB07_Leo"
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

    # Load checkpoint if resuming training
    if resume_from_checkpoint:
        try:
            trainer.load_checkpoint(checkpoint_path)
            print(f"Successfully resumed from checkpoint")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting fresh training...")
            # Reset start_epoch to 0 if checkpoint loading fails
            trainer.start_epoch = 0

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    # Finish wandb wandb_run
    wandb_run.finish()

if __name__ == '__main__':
    wandb.login(key="9d7294877630de627f0413ca39aebd4c9a387e50")
    if use_sweep:
        # Initialize the sweep
        sweep_id = wandb.sweep(sweep_config, project= hyperparam_config['wandb_project_name'])
        # Start the sweep agent
        wandb.agent(sweep_id, function=train)
    else:
        # For a single training wandb_run, call train() directly
        train()

