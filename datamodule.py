from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import pytorch_lightning as pl
import torchmetrics
from dataset import TractogramDataset, collate_fn
from val_dataset import ValTractogramDataset, collate_fn_fast


class TractogramDM(pl.LightningDataModule):
    def __init__(self, data_dir, label_dir, batch_size, num_workers, mode, binarize=True):
        super().__init__()
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_end = "VAL"
        self.train_end = "TRAIN"
        self.test_end = "TEST"
        self.mode = mode
        self.binarize = binarize

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.train_ds = TractogramDataset(
            os.path.join(self.data_dir, self.train_end),
            os.path.join(self.label_dir, self.train_end),
            mode=self.mode,
        )
        self.val_ds = ValTractogramDataset(
            os.path.join(self.data_dir, self.val_end),
            os.path.join(self.label_dir, self.val_end),
            mode=self.mode,
            binarize=self.binarize,
        )
        self.test_ds = ValTractogramDataset(
            os.path.join(self.data_dir, self.test_end),
            os.path.join(self.label_dir, self.test_end),
            mode=self.mode,
            binarize=self.binarize,
        )
        print(len(self.train_ds), len(self.val_ds))

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=32,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn_fast,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=2,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn_fast,
        )
