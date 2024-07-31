import torch
import copy
from torch.utils.data import Dataset, DataLoader
import os
import h5py
import numpy as np
import pandas as pd
import json
from utils import (
    scale_data_min_max,
    filter_data_labels,
    resample_3d_sequence_batch_torch,
)


class ValTractogramDataset(Dataset):
    def __init__(
        self,
        data_dir,
        label_dir,
        mode="sift",
        binarize=True,
    ):

        self.files = []
        self.labels = []
        self.num_files = 0
        print(os.listdir(data_dir))
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
            self.num_files += len(file_list)
        self.lengths = []
        self.total_length = 0
        # Calculate the total number of streamlines
        self.total_length = 1000*len(self.files)
        self.lengths = [1000]*len(self.files)
        print("Total length: ", self.total_length)
        print("Number of files: ", self.num_files)
        self.prev_file_idx = None
        self.mode = mode
        self.mean = torch.tensor([-0.2778, -25.4457, 14.5391]).view(1, 1, -1)
        self.std = torch.tensor([28.1488, 35.9286, 22.5833]).view(1, 1, -1)
        self.bin = binarize

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        file_idx = idx
        if self.prev_file_idx != file_idx:
            self.label_df = pd.read_csv(self.labels[file_idx], index_col=0)
            with h5py.File(self.files[file_idx], "r") as f:
                self.data = f["streamlines"][()]

        subject = self.labels[file_idx].split("/")[-1].split("_")[0]
        # take out the corresponding labels: 0 for negative, 1 for union positive, 2 for intersection positive
        neg_threshold = 1#1e-6 
        if self.mode == "sift":
            labels = self.label_df[
                "sift_acc"
            ].values  # Keeping the labels as float for threshold comparisons
            data = self.data
            if self.bin:
                data, labels = filter_data_labels(data, labels, neg_threshold)
                # Convert the boolean labels to the long type tensor required for training
                labels = torch.tensor(labels >= 1).long()
            else:
                labels = torch.tensor(labels).float()

        elif self.mode == "intersection":
            labels = self.label_df.values.values
            labels_cond = (labels >= 1)[
                :, -2:
            ]  # Change the condition to mask for the last two columns
            labels_prod = labels_cond.prod(axis=1)
            data = self.data
            if self.bin:
                data, labels_prod = filter_data_labels(data, labels_prod, neg_threshold)
                labels = torch.tensor(labels_prod).long()
            else:
                labels = torch.tensor(labels_prod).float()

        elif self.mode == "commit":
            labels = self.label_df[
                "commit_acc"
            ].values  # Keeping the labels as float for threshold comparisons
            data = self.data

            if self.bin:
                data, labels = filter_data_labels(data, labels, neg_threshold)
                labels = torch.tensor(labels >= 1).long()
            else:
                labels = torch.tensor(labels).float()
        self.prev_file_idx = file_idx
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

        # Normalize only non-zero elements
        data = torch.where(
            nonzero_mask,
            (data-self.mean)/self.std,  # If the condition is True, scale the data
            data,  # If the condition is False, retain original values (zero-padded elements)
        )
        # Resample the data to 23 points
        data = resample_3d_sequence_batch_torch(data, 23)
        return data, labels, subject


def collate_fn_fast(batch):
    # Separate the streamlines and the labels
    streamlines, labels, subjects = zip(*batch)

    max_len = max(item.shape[1] for item in streamlines)
    # Use the pad_sequence function from PyTorch to pad all tensors to the max length
    streamlines = [
        torch.nn.functional.pad(item, (0, 0, 0, max_len - item.shape[1]))
        for item in streamlines
    ]

    # Concatenate all tensors in the batch
    streamlines = torch.cat(streamlines, dim=0)

    # Concatenate labels using torch.stack for more efficient tensor operation
    labels = torch.cat(labels)
    return streamlines, labels


