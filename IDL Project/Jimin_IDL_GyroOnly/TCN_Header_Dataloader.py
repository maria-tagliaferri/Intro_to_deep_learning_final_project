# ICORR_Header_Dataloader.py
import os
import numpy as np
import pandas as pd
from scipy import signal
import torch
from torch.utils.data import Subset
import re

class DataHandler:
    def __init__(self, data_root, hyperparam_config, pretrained_model_path=None):

        self.data_root = data_root
        self.window_size = hyperparam_config["window_size"]
        self.batch_size = hyperparam_config["batch_size"]
        self.num_workers = hyperparam_config["number_of_workers"]
        self.validation_split = hyperparam_config["validation_split"]
        self.dataset_proportion = hyperparam_config["dataset_proportion"]
        self.transfer_learning = hyperparam_config["transfer_learning"]
        self.pretrained_model_path = pretrained_model_path
            
        # Placeholder for data
        self.train_data = None
        self.test_data = None

        # Initialize mean and std attributes
        self.input_mean = None
        self.input_std = None
        self.label_mean = None
        self.label_std = None

        # (For transfer learning) Load saved mean and std for normalization
        if self.pretrained_model_path is not None:
            self.input_mean = np.load(os.path.join(self.pretrained_model_path, 'input_mean.npy'))
            self.input_std = np.load(os.path.join(self.pretrained_model_path, 'input_std.npy'))
            self.label_mean = np.load(os.path.join(self.pretrained_model_path, 'label_mean.npy'))
            self.label_std = np.load(os.path.join(self.pretrained_model_path, 'label_std.npy'))
        
    def load_data(self, train_data_partition, speed_condition, task_condition, test_data_partition, test_data_partition_2=None, test_data_partition_3=None):

        # Load training data (including data for validation split)
        print("\n...Loading training data...\n")        
        self.train_data = LoadData(
            root=self.data_root,
            partitions= train_data_partition,
            speed_conditions= speed_condition,
            task_conditions= task_condition,
            window_size=self.window_size,
            data_type = "train_data" if self.transfer_learning == False else "train_data_tranfer_learning",
            dataset_proportion=self.dataset_proportion,
            input_mean= self.input_mean,
            input_std= self.input_std,
            label_mean= self.label_mean,
            label_std= self.label_std
        )
        
        # Retrieve mean and std from training data
        self.input_mean = self.train_data.input_mean
        self.input_std = self.train_data.input_std
        self.label_mean = self.train_data.label_mean
        self.label_std = self.train_data.label_std
        
        # Load test data using training mean and std
        print("\n...Loading test data...\n")        
        self.test_data = LoadData(
            root=self.data_root,
            partitions= test_data_partition,
            speed_conditions= speed_condition,  # Use the same conditions as training data
            task_conditions= task_condition,  # Use the same conditions as training data
            window_size=self.window_size,
            data_type = "test_data",
            dataset_proportion=self.dataset_proportion
        )
        # Replace the mean and std of test data with that of training data
        self.test_data.input_mean = self.input_mean
        self.test_data.input_std = self.input_std
        self.test_data.label_mean = self.label_mean
        self.test_data.label_std = self.label_std
        
    def save_mean_std(self, save_dir):
        # Save mean and std to .npy files
        np.save(os.path.join(save_dir, 'input_mean.npy'), self.input_mean)
        np.save(os.path.join(save_dir, 'input_std.npy'), self.input_std)
        np.save(os.path.join(save_dir, 'label_mean.npy'), self.label_mean)
        np.save(os.path.join(save_dir, 'label_std.npy'), self.label_std)
     
    def get_train_val_indices(self):
        # Randomly split indices for training and validation
        if len(self.train_data.subject_data_length) == 1:
            # Special case: only one subject
            total_length = self.train_data.subject_data_length[0] - self.window_size + 1
            indices = list(range(total_length))
            split_point = int(np.floor(total_length * (1 - self.validation_split)))
            
            # Shuffle indices before splitting
            np.random.shuffle(indices)
            train_indices = indices[:split_point]
            val_indices = indices[split_point:]
            print(f"\nSingle subject detected. Random split: {len(train_indices)} train, {len(val_indices)} validation samples")
        else:
            # Multiple subjects: leave one subject out
            leave_one_subject_out = np.random.randint(0, len(self.train_data.subject_data_length))
            print(f"\nLeave out subject: {leave_one_subject_out+1}")

            leave_out_start = sum(self.train_data.subject_data_length[:leave_one_subject_out])
            leave_out_end = sum(self.train_data.subject_data_length[:leave_one_subject_out+1]) - self.window_size + 1
            total_length = sum(self.train_data.subject_data_length) - self.window_size + 1
            train_indices = list(range(0, leave_out_start)) + list(range(leave_out_end, total_length))
            val_indices = list(range(leave_out_start, leave_out_end))

        return train_indices, val_indices
    
    def create_dataloaders(self, train_indices=None, val_indices=None, test_indices=None):
        # Create subsets and DataLoaders
        if train_indices is not None and val_indices is not None:
            train_subset = Subset(self.train_data, train_indices)
            val_subset = Subset(self.train_data, val_indices)
            
            train_loader = torch.utils.data.DataLoader(
                dataset=train_subset,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                pin_memory=True,
                shuffle=True
            )
            
            val_loader = torch.utils.data.DataLoader(
                dataset=val_subset,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                pin_memory=True,
                shuffle=False
            )
            return train_loader, val_loader
        else:
            if test_indices == 1:
                # Create Test DataLoader
                test_loader = torch.utils.data.DataLoader(
                    dataset=self.test_data,
                    num_workers=self.num_workers,
                    batch_size=self.batch_size,
                    pin_memory=True,
                    shuffle=False
                )
            
            return test_loader

# Dataset class to load data
class LoadData(torch.utils.data.Dataset):
    def __init__(self, root, partitions, speed_conditions, task_conditions, window_size, data_type, dataset_proportion=None, input_mean=None, input_std=None, label_mean=None, label_std=None, normalize = True):
        self.window_size = window_size
        self.input_list = []
        self.label_list = []
        self.data_type = data_type
        self.normalize = normalize
        self.dataset_proportion = dataset_proportion
        self.subject_data_length = []
        
        for subject_num, subject in enumerate(partitions):  # Multiple partitions
            self.subject_data_length.append(0) # Append 0 for each subject
            for task in os.listdir(os.path.join(root, subject)):
                # Task 조건 필터링
                if task not in task_conditions:
                    continue
                    
                task_path = os.path.join(root, subject, task)
                
                def process_trial_data(input_file_dir, label_file_dir, print_info):
                    """공통 데이터 처리 함수"""
                    input_file_names = sorted(os.listdir(input_file_dir))
                    label_file_names = sorted(os.listdir(label_file_dir))
                    print(print_info)

                    # Load and concatenate all input data files
                    input_buffer_R = None
                    input_buffer_L = None

                    for i, name in enumerate(input_file_names):
                        csv_path = os.path.join(input_file_dir, name)

                        # Extract IMU data from input file
                        if 'imu' in name.lower():
                            # Right side
                            input_df_R = pd.read_csv(csv_path, delimiter=',', on_bad_lines='skip')[[
                                # 'Pelvis_Acc_X', 'Pelvis_Acc_Y', 'Pelvis_Acc_Z', 'Pelvis_Gyr_X', 'Pelvis_Gyr_Y', 'Pelvis_Gyr_Z',
                                # 'Thigh_R_Acc_X', 'Thigh_R_Acc_Y', 'Thigh_R_Acc_Z'
                                'Thigh_R_Gyr_X', 'Thigh_R_Gyr_Y', 'Thigh_R_Gyr_Z'
                                ]].values
                            # Left side
                            input_df_L = pd.read_csv(csv_path, delimiter=',', on_bad_lines='skip')[[
                                # 'Pelvis_Acc_X', 'Pelvis_Acc_Y', 'Pelvis_Acc_Z', 'Pelvis_Gyr_X', 'Pelvis_Gyr_Y', 'Pelvis_Gyr_Z',
                                # 'Thigh_L_Acc_X', 'Thigh_L_Acc_Y', 'Thigh_L_Acc_Z'
                                'Thigh_L_Gyr_X', 'Thigh_L_Gyr_Y', 'Thigh_L_Gyr_Z'
                                ]].values
                            # Left side: Flip the signs of Thigh_L_Acc_Y, Thigh_L_Gyr_X, Thigh_L_Gyr_Z
                            # input_df_L[:, [1, 3, 5]] *= -1
                            input_df_L[:, [0, 2]] *= -1
                            
                            print(f'input df_R shape is this : {input_df_R.shape}')
                            print(f'input df_L shape is this : {input_df_L.shape}')

                            # Horizontally stack all input data (only for IMU files)
                            if input_buffer_R is None:    input_buffer_R = input_df_R
                            else:   input_buffer_R = np.hstack((input_buffer_R, input_df_R))
                            if input_buffer_L is None:    input_buffer_L = input_df_L
                            else:   input_buffer_L = np.hstack((input_buffer_L, input_df_L))
                            print(f"\tinput file {i+1} loaded: ", name)
                            print(f'input df_R shape is this : {input_df_R.shape}')
                            print(f'input df_L shape is this : {input_df_L.shape}')
                        else:
                            print(f"\tskipping non-IMU file: ", name)

                        # # Extract motor data from input file
                        # elif 'motor' in name.lower():
                        #     # Right side
                        #     input_df_R = pd.read_csv(csv_path, delimiter=',', on_bad_lines='skip')[[
                        #         'mtr_pos_R', 'mtr_vel_R'
                        #         ]].values
                        #     # Left side
                        #     input_df_L = pd.read_csv(csv_path, delimiter=',', on_bad_lines='skip')[[
                        #         'mtr_pos_L', 'mtr_vel_L'
                        #         ]].values
                        #     # Left side: Flip the signs of mtr_pos_L, mtr_vel_L
                        #     input_df_L[:, [0, 1]] *= -1
                        
                        # if input_buffer_R is None:    input_buffer_R = input_df_R
                        # else:   input_buffer_R = np.hstack((input_buffer_R, input_df_R))
                        # if input_buffer_L is None:    input_buffer_L = input_df_L
                        # else:   input_buffer_L = np.hstack((input_buffer_L, input_df_L))
                        # print(f"\tinput file {i+1} loaded: ", name)
                        
                        
                    # Segment train data and test data based on dataset_proportion
                    input_time_sec = int(input_buffer_R.shape[0]/100) # Extract recording time from input file by dividing 100 Hz
                    if self.data_type == "train_data":
                        input_buffer_R = input_buffer_R[:int(input_buffer_R.shape[0]* self.dataset_proportion), :] # Use (dataset_proportion)% of the data for training
                        input_buffer_L = input_buffer_L[:int(input_buffer_L.shape[0]* self.dataset_proportion), :]
                    elif self.data_type == "train_data_tranfer_learning":
                        input_buffer_R = input_buffer_R[:int(input_buffer_R.shape[0]* self.dataset_proportion), :]
                        input_buffer_L = input_buffer_L[:int(input_buffer_L.shape[0]* self.dataset_proportion), :]
                    elif self.data_type == "test_data":
                        input_buffer_R = input_buffer_R[:int(input_buffer_R.shape[0]* self.dataset_proportion), :] # Use 10% of the data for testing
                        input_buffer_L = input_buffer_L[:int(input_buffer_L.shape[0]* self.dataset_proportion), :]
                    
                    R_side_first = np.random.randint(0, 2) # Randomly select right or left side as the first column
                    # Randomly select right or left side as the first column
                    if R_side_first == 0:
                        input_buffer = np.vstack((input_buffer_R, input_buffer_L))
                    else:
                        input_buffer = np.vstack((input_buffer_L, input_buffer_R))

                    print(f'input buffer shape is this : {input_buffer.shape}')

                    # Load and label data file (Vicon data)
                    label_buffer = self.load_vicon_hip_moment_data(label_file_dir, label_file_names[0], input_time_sec) # Extract recording time from input file by dividing 100 Hz
                    print(f"\tlabel file loaded: ", label_file_names)
                    
                    # # 0mps 조건일 때 label을 모두 0으로 설정
                    # if '0mps' in label_file_dir or '0mps' in print_info:
                    #     label_buffer[:] = 0
                    #     print(f"\t[ZERO_LABELING] File: {label_file_dir}")
                    #     print(f"\t[ZERO_LABELING] Info: {print_info.strip()}")
                    
                    # 정확히 "0mps"만 매치 (앞뒤에 문자가 없거나 구분자가 있는 경우)
                    if re.search(r'\b0mps\b', label_file_dir) or re.search(r'\b0mps\b', print_info):
                        label_buffer[:] = 0
                        print(f"\t[DEBUG] 0mps detected - labels set to zero")
                        print(f"\t[ZERO_LABELING] File: {label_file_dir}")
                        print(f"\t[ZERO_LABELING] Info: {print_info.strip()}")
                        

                    # label_buffer_R = label_buffer[:, 1].reshape(-1, 1) # Right hip moment
                    # label_buffer_L = label_buffer[:, 0].reshape(-1, 1) # Left hip moment
                    
                    label_buffer_R = label_buffer[:, 0].reshape(-1, 1) # Right hip moment
                    label_buffer_L = label_buffer[:, 1].reshape(-1, 1) # Left hip moment

                    # Segment train data and test data based on dataset_proportion
                    if self.data_type == "train_data":
                        label_buffer_R = label_buffer_R[:int(label_buffer_R.shape[0]* self.dataset_proportion), :] # Use (dataset_proportion)% of the data for training
                        label_buffer_L = label_buffer_L[:int(label_buffer_L.shape[0]* self.dataset_proportion), :]
                    elif self.data_type == "train_data_tranfer_learning":
                        label_buffer_R = label_buffer_R[:int(label_buffer_R.shape[0]* self.dataset_proportion), :]
                        label_buffer_L = label_buffer_L[:int(label_buffer_L.shape[0]* self.dataset_proportion), :]
                    elif self.data_type == "test_data":
                        label_buffer_R = label_buffer_R[:int(label_buffer_R.shape[0]* self.dataset_proportion), :] # Use 10% of the data for testing
                        label_buffer_L = label_buffer_L[:int(label_buffer_L.shape[0]* self.dataset_proportion), :]
                    
                    # Randomly select right or left side as the first column
                    if R_side_first == 0:
                        label_buffer = np.vstack((label_buffer_R, label_buffer_L))
                    else:
                        label_buffer = np.vstack((label_buffer_L, label_buffer_R))
                    
                    print(f'input buffer shape is this : {input_buffer.shape}, label buffer shape: {label_buffer.shape}')
                    self.input_list.append(input_buffer)
                    self.label_list.append(label_buffer)
                    self.subject_data_length[subject_num] += input_buffer.shape[0]

                if task == 'LG':
                    # LG의 경우: LG > Speed > Trial 구조
                    for speed in os.listdir(task_path):
                        # Speed 조건 필터링
                        if speed not in speed_conditions:
                            continue
                            
                        speed_path = os.path.join(task_path, speed)
                        for trial in os.listdir(speed_path):
                            input_file_dir = os.path.join(speed_path, trial, 'Input')
                            label_file_dir = os.path.join(speed_path, trial, 'Label')
                            process_trial_data(input_file_dir, label_file_dir, f"\n {subject} {task} {speed} {trial}")

                elif task in ['RA', 'RD']:
                    # RA/RD의 경우: RA/RD > Degree > Speed > Trial 구조
                    for degree in os.listdir(task_path):
                        degree_path = os.path.join(task_path, degree)
                        for speed in os.listdir(degree_path):
                            # Speed 조건 필터링
                            if speed not in speed_conditions:
                                continue
                                
                            speed_path = os.path.join(degree_path, speed)
                            for trial in os.listdir(speed_path):
                                input_file_dir = os.path.join(speed_path, trial, 'Input')
                                label_file_dir = os.path.join(speed_path, trial, 'Label')
                                process_trial_data(input_file_dir, label_file_dir, f"\n {subject} {task} {degree} {speed} {trial}")
                else:
                    # 기존 구조 (backward compatibility)
                    for trial in os.listdir(task_path):
                        input_file_dir = os.path.join(task_path, trial, 'Input')
                        label_file_dir = os.path.join(task_path, trial, 'Label')
                        process_trial_data(input_file_dir, label_file_dir, f"\n {subject} {task} {trial}")
            
            print("\nsubject data legnth: ", self.subject_data_length[subject_num])
        
        
        # Concatenate all data
        self.input = np.concatenate(self.input_list, axis=0)
        self.label = np.concatenate(self.label_list, axis=0)

        print(f'\ninput {self.data_type} dataset size: ', np.shape(self.input))
        print(f'label {self.data_type} dataset size: ', np.shape(self.label))

        self.length = len(self.input) - self.window_size + 1
        print(f"Total {self.data_type} sequences: ", self.length)
        
        # Calculate mean and std using the entire dataset
        self.input_mean = np.mean(self.input, axis=0)
        self.input_std = np.std(self.input, axis=0) + 1e-8

        self.label_mean = np.mean(self.label, axis=0)
        self.label_std = np.std(self.label, axis=0) + 1e-8

        # Override with provided mean and std if given (for transfer learning)
        if input_mean is not None:
            self.input_mean = input_mean
        if input_std is not None:
            self.input_std = input_std
        if label_mean is not None:
            self.label_mean = label_mean
        if label_std is not None:
            self.label_std = label_std

    def load_vicon_hip_moment_data(self, label_file_dir, label_file_name, record_time_sec):
        output_df = pd.read_csv(os.path.join(label_file_dir, label_file_name),
                              delimiter=',',
                              skiprows=lambda x: x in range(0, record_time_sec*1000 + 10), # skip force plate data and 10 rows of header
                              encoding_errors='ignore') # Add this line to ignore encoding errors
        output_df = output_df.fillna(0) # fill NaN with 0
        output_buffer = output_df.values[:, [54, 6]]/1000 # 54 for right hip, 6 for left hip, divide by 1000 to convert to Nm
        output_buffer[:, 0] *= -1 # Flip the sign of right hip moment to make it Flex. = +
        output_buffer = self.lowpass_filter(output_buffer, order=4, cutoff_freq=6, sampling_freq=100)
        return output_buffer

    def lowpass_filter(self, data, order=4, cutoff_freq=6, sampling_freq=100):
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

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        windows_input = self.input[ind: ind + self.window_size]     # Shape: (window_size, input_size)
        # Normalize the data using stored mean and std
        if self.normalize == True:
            windows_input = (windows_input - self.input_mean) / self.input_std

        # Convert to tensor without flattening
        window_input = torch.FloatTensor(windows_input).T           # Shape: (input_size, window_size)
        # print(f"window_input shape: {window_input.shape}")

        # Get the target joint moments at the last time point in the window
        target_label = self.label[ind + self.window_size - 1]       # Shape: (output_size)
        
        # Normalize the target joint moments
        if self.normalize == True:
            target_label = (target_label - self.label_mean) / self.label_std
            
        window_label = torch.FloatTensor(target_label) # Shape: (output_size), consider putting .T when output_size > 1
        # print(f"window_label shape: {window_label.shape}")
        
        return window_input, window_label