import torch
from torch.utils.data import Dataset, DataLoader
import os
import h5py
import numpy as np
import pandas as pd
import random
import json
from utils import (
    scale_data_min_max,
    filter_data_labels,
    resample_3d_sequence_batch_torch,
)


class TractogramDataset(Dataset):
    def __init__(
        self,
        data_dir,
        label_dir,
        mode="sift",
    ):
        self.files = []
        self.labels = []
        self.num_streamlines = 400


        for folder in os.listdir(data_dir):
            file_list = os.listdir(os.path.join(data_dir, folder))
            self.files += [
                os.path.join(data_dir, folder, f)
                for f in file_list
                if f.endswith(".hdf5")
            ]
            self.labels += [
                os.path.join(label_dir, folder, f.replace(".hdf5", ".csv"))
                for f in file_list
                if f.endswith(".hdf5")
            ]
        # Calculate the total number of streamlines
        self.lengths = [1000] * len(self.files)
        self.total_length = 1000* len(self.files)
        self.mode = mode
      
        self.mean = torch.tensor([-0.2778, -25.4457, 14.5391]).view(1, 1, -1)
        self.std = torch.tensor([28.1488, 35.9286, 22.5833]).view(1, 1, -1)

    def __len__(self):
        return int(self.total_length)

    def __getitem__(self, idx):
        # Find the file that contains the streamline for the given index
        file_idx = np.searchsorted(np.cumsum(self.lengths), idx + 1)
        # Find the index of the streamline within the file
        streamline_idx = idx - (
            np.cumsum(self.lengths)[file_idx - 1] if file_idx > 0 else 0
        )
        # Load the corresponding label
        label_df = pd.read_csv(self.labels[file_idx], index_col=0)
        subject = self.labels[file_idx].split("/")[-1].split("_")[0]
        if self.mode == "sift":
            # find indices of all streamlines with acc_rate >= 1
            pos_indices = np.where((label_df["sift_acc"] >= 1))[0]
            np.random.shuffle(pos_indices)
            pos_indices = pos_indices[: self.num_streamlines]
            # load extra at most self.num_streamlines randomly selected streamlines
            neg_indices = np.where((label_df["sift_acc"] < 1))[0]
            neg_indices = np.random.choice(
                neg_indices, size=max(len(pos_indices), 1), replace=False
            )
            # concatenate the two
            streamline_indices = np.concatenate((pos_indices, neg_indices))
            # sort the indices
            streamline_indices = np.sort(streamline_indices)

            # take out the corresponding labels: 0 for negative, 1 for union positive, 2 for intersection positive
            labels = (label_df.iloc[streamline_indices]["sift_acc"] >= 1).values.astype(
                np.int32
            )

            labels = torch.tensor(labels).long()

        elif self.mode == "commit":
            # find indices of all streamlines with acc_rate >= 1
            pos_indices = np.where((label_df["commit_acc"] >= 1))[0]
            np.random.shuffle(pos_indices)
            pos_indices = pos_indices[: self.num_streamlines]
            # load extra at most self.num_streamlines randomly selected streamlines
            neg_indices = np.where((label_df["commit_acc"] < 1))[0]
            neg_indices = np.random.choice(
                neg_indices, size=max(len(pos_indices), 1), replace=False
            )
            # concatenate the two
            streamline_indices = np.concatenate((pos_indices, neg_indices))
            # sort the indices
            streamline_indices = np.sort(streamline_indices)

            # take out the corresponding labels: 0 for negative, 1 for union positive, 2 for intersection positive
            labels = (
                label_df.iloc[streamline_indices]["commit_acc"] >= 1
            ).values.astype(np.int32)
            labels = torch.tensor(labels).long()

        elif self.mode == "intersection":
            # find indices of all streamlines with acc_rate >= 1
            pos_indices = np.where(
                (label_df["sift_acc"] >= 1) & (label_df["commit_acc"] >= 1)
            )[0]
            np.random.shuffle(pos_indices)
            pos_indices = pos_indices[: self.num_streamlines]
            # load extra at most self.num_streamlines randomly selected streamlines
            neg_indices = np.where(
                (label_df["sift_acc"] < 1) | (label_df["commit_acc"] < 1)
            )[0]
            neg_indices = np.random.choice(
                neg_indices, size=max(len(pos_indices), 1), replace=False
            )
            # concatenate the two
            streamline_indices = np.concatenate((pos_indices, neg_indices))
            # sort the indices
            streamline_indices = np.sort(streamline_indices)

            # take out the corresponding labels: 0 for negative, 1 for union positive, 2 for intersection positive
            labels = (label_df.iloc[streamline_indices] >= 1).values.astype(np.int32)

            # If we want to distinguish between commit and sift positive
            # If we want to combine the two to intersection only
            labels = torch.tensor(labels[:, -2:].prod(axis=1)).long()


        # Load the streamline
        with h5py.File(self.files[file_idx], "r") as f:
            data = f["streamlines"][streamline_indices]
            # check if any nan values in the data
        if np.isnan(data).any():
            print("nan values in data")
            print(self.files[file_idx])
            print(streamline_idx)

        # check if any nan values in the labels
        if np.isnan(labels).any():
            print("nan values in labels")
            print(self.labels[file_idx])
            print(streamline_idx)
            import sys

            sys.exit()
        data = torch.from_numpy(data)

        nonzero_mask = data != 0

        data = torch.where(
            nonzero_mask,
            (data-self.mean)/self.std,
            data,  # If the condition is False, retain original values (zero-padded elements)
        )
        # Resample the data to 23 points
        data = resample_3d_sequence_batch_torch(data, 23)


        return data, labels, [subject]*len(labels)


# Make collate to pad the batch
def collate_fn(batch, max_len=100):
    # Separate the streamlines and the labels
    streamlines, labels, subjects = zip(*batch)

    max_len = max(item.shape[1] for item in streamlines)

    # Pad each tensor in the batch
    streamlines = [
        torch.nn.functional.pad(item, (0, 0, 0, max_len - item.shape[1]), value=0.0)
        for item in streamlines
    ]

    # Concatenate all tensors in the batch
    batch = torch.cat(streamlines, dim=0)
    batch = batch[:, :max_len, :]
    
    # concatenate list of subjects
    subjects = [item for sublist in subjects for item in sublist]

    return batch, torch.cat(labels)



