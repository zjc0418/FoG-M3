import os
import random
import tqdm
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import torch
from torch.utils.data import Dataset

def butterworth_lowpass_filter(data, cutoff, fs, order):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def read_data(folder, stride=50, i_mix=True, i_filter=True, fs=64, cutoff=20.0, order=3, mode="pre"):
    window_x, window_y, window_ysum = [], [], []

    all_classes = [0, 1, 2]
    txt_name = os.listdir(folder)

    for file_name in tqdm.tqdm(txt_name, desc='Processing files'):
        file_path = os.path.join(folder, file_name)
        columns = ['Time', 'Ankle_hf', 'Ankle_v', 'Ankle_hl', 'Upper_leg_hf', 'Upper_leg_v', 
                   'Upper_leg_hl', 'Trunk_hf', 'Trunk_v', 'Trunk_hl', 'label']
        df = pd.read_csv(file_path, sep=' ', header=None, names=columns)
        
        # Apply filtering if enabled
        if i_filter:
            for col in ['Ankle_hf', 'Ankle_v', 'Ankle_hl', 'Upper_leg_hf', 'Upper_leg_v', 
                        'Upper_leg_hl', 'Trunk_hf', 'Trunk_v', 'Trunk_hl']:
                df[col] = butterworth_lowpass_filter(df[col], cutoff, fs, order)
        
        df['i_valid'] = 0
        index_b = []
        for i in range(1, len(df) - 1):
            if df.loc[i-1, 'label'] == 1 and df.loc[i, 'label'] == 2 and df.loc[i+1, 'label'] == 2:
                index_b.append(i)

        for idx, row in df.iterrows():
            if row['label'] in [0, 1]: 
                closest_index = min(index_b, key=lambda x: abs(x - idx)) if index_b else None
                if closest_index is not None and abs(closest_index - idx) <= 576:
                    df.at[idx, 'i_valid'] = 1
            elif row['label'] == 2:  # For label 2
                closest_index = min(index_b, key=lambda x: abs(x - idx)) if index_b else None
                if closest_index is not None and abs(idx - closest_index) < 64:
                    df.at[idx, 'i_valid'] = 1

        # One-hot encoding for the labels
        one_hot_encoded = pd.get_dummies(df['label'], prefix='label')
        one_hot_encoded = one_hot_encoded.reindex(columns=['label_' + str(c) for c in all_classes], fill_value=0).astype(int)

        # Combine one-hot encoded columns with the original dataframe
        df = pd.concat([df, one_hot_encoded], axis=1)

        # Drop original label column
        df.drop('label', axis=1, inplace=True)

        # Windowing
        total_length = len(df)
        index = 0
        while index + stride <= total_length:
            window = df.iloc[index: index + stride]
            windowx = window.loc[:, ['Ankle_hf', 'Ankle_v', 'Ankle_hl', 'Upper_leg_hf', 'Upper_leg_v', 
                                     'Upper_leg_hl', 'Trunk_hf', 'Trunk_v', 'Trunk_hl', 'i_valid']].reset_index(drop=True)
            windowy = window.loc[:, ["label_0", "label_1", "label_2"]].reset_index(drop=True)

            i_label_sum = [windowy[col].sum() for col in ["label_0", "label_1", "label_2"]]

            if i_label_sum[0] == 0 and len(windowx) == stride:
                window_x.append(windowx)
                window_y.append(windowy)
                window_ysum.append(i_label_sum)

            if mode == "label":
                index += stride  
            else:
                index += stride // 2

    # Convert to numpy arrays
    window_x = np.array([x.values for x in window_x], dtype=np.float32)
    window_y = np.array([y.values for y in window_y], dtype=np.int32)


    total_length = len(window_x)
    train_count = int(0.8 * total_length)
    validation_count = int(0.1 * total_length)

    train_indices = random.sample(range(total_length), train_count)
    remaining_indices = [i for i in range(total_length) if i not in train_indices]
    validation_indices = random.sample(remaining_indices, validation_count)
    test_indices = [i for i in remaining_indices if i not in validation_indices]  # Not shuffled

    if i_mix:
        train_set = [(window_x[i], window_y[i], window_ysum[i]) for i in train_indices]
    else:
        train_set = [(window_x[i], window_y[i]) for i in train_indices]

    validation_set = [(window_x[i], window_y[i]) for i in validation_indices]
    test_set = [(window_x[i], window_y[i]) for i in test_indices]
    total_set = [(window_x[i], window_y[i]) for i in range(total_length)]

    return total_set, train_set, validation_set, test_set


class Daphnetwindow(Dataset):
    def __init__(self, set, mode = "train"):
        super().__init__()
        self.set = set
        self.mode = mode
    def __getitem__(self, index):
        if self.mode == "pre":
            data_x, data_y = self.set[index]
            x = torch.tensor(data_x[:, :9], dtype=torch.float32).mT
            y = torch.tensor(data_y[:, 1:], dtype=torch.long)
            return x, y
        if self.mode == "label":
            data_x, data_y = self.set[index]
            x = torch.tensor(data_x[:, :9], dtype=torch.float32).mT
            y = torch.tensor(data_y[:, 1:], dtype=torch.long)
            v = torch.tensor(data_x[:, 9], dtype=torch.float32)
            return x,y,v
        else:
            data_x, data_y, data_label = self.set[index]
            x = torch.tensor(data_x[:, :9], dtype=torch.float32).mT
            y = torch.tensor(data_y[:, 1:], dtype=torch.long)
            z = torch.tensor(data_label, dtype=torch.long)
            return x, y, z
        
    def __len__(self):
        return len(self.set)

