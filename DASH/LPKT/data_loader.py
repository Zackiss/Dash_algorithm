import pandas as pd
import torch
from torch.utils.data import Dataset

from DASH.encoder import Encoder


class StudentSequenceDataset(Dataset):
    def __init__(self, csv_path, max_length, encoder: Encoder):
        self.max_length = max_length
        self.data = pd.read_csv(csv_path)
        self.user_ids = self.data['user_id'].unique()
        self.encoder = encoder

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        user_data = self.data[self.data['user_id'] == user_id]

        # Retrieve relevant columns
        problem_ids = user_data['problem_id'].values
        outcomes = user_data['outcome'].values
        interval_times = user_data['interval_time'].values
        time_spent = user_data['time_spent'].values

        # Pad or truncate sequences to max_length
        seq_length = min(len(problem_ids), self.max_length)
        problem_ids = problem_ids[:seq_length]
        outcomes = outcomes[:seq_length]
        interval_times = interval_times[:seq_length]
        time_spent = time_spent[:seq_length]

        # Convert to tensors
        return self.encoder.encode_deck(
            problem_ids=problem_ids,
            outcomes=outcomes,
            interval_times=interval_times,
            times_spent=time_spent,
        )
