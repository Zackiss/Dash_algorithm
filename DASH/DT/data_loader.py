import random
import time
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

from DASH.LPKT import LPKT
from DASH.LPKT.data_loader import StudentSequenceDataset
from torch.utils.data import DataLoader


class TrajectoryDataset(Dataset):
    def __init__(self, sequence_dataset: StudentSequenceDataset, context_len, gamma):
        self.sequence_dataset = sequence_dataset
        self.context_len = context_len
        self.gamma = gamma

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        time_steps, actions, traj_mask = self.get_trajectory(idx)
        return time_steps, actions, traj_mask

    def get_trajectory(self, idx):
        # Get the sequence data from the StudentSequenceDataset
        sequence_data = self.sequence_dataset[idx]
        time_steps = torch.arange(start=0, end=self.context_len, step=1)

        # Retrieve the relevant tensors from the sequence data
        problem_ids, _, time_spent, interval_times = sequence_data
        seq_length = len(problem_ids)

        if seq_length >= self.context_len:
            # Slice the tensors
            # Sample random index to slice sequence
            si = random.randint(0, seq_length - self.context_len)
            actions = [
                problem_ids[si: si + self.context_len],
                # (time_n * self.context_len)
                time_spent[si: si + self.context_len],
                interval_times[si: si + self.context_len]
            ]
            # All ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)
        else:
            # Pad the tensors with zeros
            padding_len = self.context_len - seq_length
            actions = [
                torch.cat([
                    problem_ids, torch.zeros((padding_len,) + problem_ids.shape[1:], dtype=problem_ids.dtype)
                ], dim=0),
                torch.cat([
                    time_spent, torch.zeros((padding_len,) + time_spent.shape[1:], dtype=time_spent.dtype)
                ], dim=0),
                torch.cat([
                    interval_times, torch.zeros((padding_len,) + interval_times.shape[1:], dtype=interval_times.dtype)
                ], dim=0)
            ]
            traj_mask = torch.cat([torch.ones(seq_length, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)],
                                  dim=0)

        return time_steps, actions, traj_mask


class TrajectoryDataLoader(DataLoader):
    def __init__(self, dataset, kt_model: LPKT, batch_size=1, shuffle=False, pin_memory=True,
                 drop_last=True, num_workers=0):
        super(TrajectoryDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, collate_fn=self.collate_fn,
            pin_memory=pin_memory, drop_last=drop_last
        )
        self.kt_model = kt_model

    def collate_fn(self, batch):
        time_steps_batch, actions_batch, traj_mask_batch = zip(*batch)
        # action_batch size:
        # batch_size * [(self.context_len, dim1),
        # (self.context_len, dim2),
        # (self.context_len, dim3)]
        states_batch, rewards_batch = self.kt_model.lpkt_net.obtain_state_from_action(actions_batch)

        # Concatenate the individual tensors from each batch
        time_steps_batch = torch.cat(time_steps_batch, dim=0)
        traj_mask_batch = torch.cat(traj_mask_batch, dim=0)
        actions_batch = torch.cat([
            torch.cat(action, dim=0) for action in zip(*actions_batch)
        ], dim=0)
        states_batch = torch.cat(states_batch, dim=0)
        rewards_batch = torch.cat(rewards_batch, dim=0)

        states_batch = torch.mean(states_batch, dim=-1)
        actions_batch = torch.mean(actions_batch, dim=-1)
        rewards_batch = torch.mean(rewards_batch, dim=(2, 3))

        return time_steps_batch, states_batch, actions_batch, rewards_batch, traj_mask_batch
        # time_steps, states, actions, returns_to_go, traj_mask
