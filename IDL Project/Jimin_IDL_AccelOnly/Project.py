
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import os
from sklearn.metrics import mean_squared_error, r2_score

def filter(data, cutoff=4, fs=100, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

lstm_path = '/Users/maria/Documents/OpenSim/4.5/Code/Python/Spring2026/prediction_estimationlstm.csv'
lstm = pd.read_csv(lstm_path, skiprows=1,  header=None)

tcn = pd.read_csv('/Users/maria/Documents/OpenSim/4.5/Code/Python/Spring2026/prediction_estimationtcn.csv', skiprows=1,  header=None)
tcn = tcn.apply(pd.to_numeric, errors='coerce').dropna()

momentlstm = filter(lstm.iloc[:, 0].to_numpy())

momenttcn = filter(tcn.iloc[:, 0].to_numpy())

ground_truth = pd.read_csv('/Users/maria/Documents/OpenSim/4.5/Code/Python/Spring2026/prediction_ground_truthlstm.csv', skiprows=1,  header=None)
ground_truth = ground_truth.apply(pd.to_numeric, errors='coerce').dropna()

ground_truth = filter(ground_truth.iloc[:, 0].to_numpy())

plt.figure(figsize=(10, 6))
plt.plot(ground_truth[0:350], label='Ground Truth', color='black', linewidth=2)
plt.plot(momentlstm[0:350], label='LSTM Prediction', color='blue', linewidth=2)
plt.plot(momenttcn[0:350], label='TCN Prediction', color='red', linewidth=2)
plt.xlabel('Time (ms)')
plt.ylabel('Hip Flexion Moment (Nm/kg)')
plt.legend()
plt.savefig('/Users/maria/Documents/OpenSim/4.5/Code/Python/Spring2026/prediction_comparison.svg', dpi=300)
plt.show()
