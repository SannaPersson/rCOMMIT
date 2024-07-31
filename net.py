import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchmetrics
import math
from torch.optim import Adam, AdamW
import torch.nn as nn



class TorchConvModel(nn.Module):
    def __init__(
        self,
        input_size=3,
        output_size=1,
        d_model=16,
        max_len=100,
        learning_rate=1e-4,
        weight_decay=0,
        padding_idx=0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.conv1 = nn.Conv1d(3, d_model, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool1d(2)
        self.fc = nn.Linear(5*d_model, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)



class ConvModel(pl.LightningModule):
    def __init__(
        self,
        input_size=3,
        output_size=1,
        d_model=16,
        max_len=100,
        learning_rate=1e-4,
        weight_decay=0,
        padding_idx=0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.model = TorchConvModel(input_size, output_size, d_model, max_len, learning_rate, weight_decay, padding_idx)

        self.loss_fn = nn.BCEWithLogitsLoss()
        # define binary accuracy
        self.accuracy = torchmetrics.classification.Accuracy(task="binary")
        self.f1_score = torchmetrics.classification.F1Score(task="binary")
        self.recall = torchmetrics.classification.Recall(task="binary")
        self.specificity = torchmetrics.classification.Specificity(task="binary")
        self.roc_auc = torchmetrics.classification.AUROC(task="binary")
        self.precision = torchmetrics.classification.Precision(task="binary")
        self.learning_rate = learning_rate

        self.weight_decay = weight_decay

        self.val_outputs = []
        self.val_y = []
        self.padding_idx = padding_idx
        self.test_outputs = []
        self.test_y = []

    def forward(self, x):
        return self.model(x)


    def _common_step(self, batch, batch_idx):
        # check data is not nan
        if torch.isnan(batch[0]).any():
            print("NAN in data")
            print(batch[0])
        if torch.isnan(batch[1]).any():
            print("NAN in labels")
            print(batch[1])
        x, y = batch
        scores = self.forward(x)
        if torch.isnan(scores).any():
            print("NAN in scores")
            print(scores)

        loss = self.loss_fn(scores.reshape(-1), y.reshape(-1).float())
        pred = scores

        pred_labels = (torch.sigmoid(pred).reshape(-1) > 0.2).long()
        accuracy = self.accuracy(pred_labels, y)
        f1_score = self.f1_score(pred_labels, y)
        # calculate f1 score manually
        precision = torch.sum(pred_labels * y) / (torch.sum(pred_labels) + 1e-6)
        recall = torch.sum(pred_labels * y) / (torch.sum(y) + 1e-6)

        f1_score = 2 * precision * recall / (precision + recall + 1e-6)
        specificity = self.specificity(pred_labels, y)
        return loss, scores, y, pred, accuracy, f1_score, recall, specificity

    def training_step(self, batch, batch_idx):
        loss, scores, y, pred, accuracy, f1_score, recall, specificity = (
            self._common_step(batch, batch_idx)
        )
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
                "train_f1_score": f1_score,
                "train_recall": recall,
                "train_specificity": specificity,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss, "scores": scores, "y": y}

    def validation_step(self, batch, batch_idx):
        loss, scores, y, pred, accuracy, f1_score, recall, specificity = (
            self._common_step(batch, batch_idx)
        )
        auc = self.roc_auc(pred, y)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_F1", f1_score, sync_dist=True)
        self.log("val_recall", recall, sync_dist=True)
        self.log("val_specificity", specificity, sync_dist=True)
        self.log("val_auc", auc, sync_dist=True)

        self.val_outputs.append(pred)
        self.val_y.append(y)

        return {"val_loss": loss, "val_f1_score": f1_score, "scores": scores, "y": y}

    def test_step(self, batch, batch_idx):
        loss, scores, y, pred, accuracy, f1_score, recall, specificity = (
            self._common_step(batch, batch_idx)
        )
        auc = self.roc_auc(pred, y)
        self.log("test_loss", loss)
        self.log("test_F1", f1_score)
        self.log("test_recall", recall)
        self.log("test_specificity", specificity)
        self.log("test_auc", auc)

        self.test_outputs.append(pred)
        self.test_y.append(y)

        return {"test_loss": loss, "test_f1_score": f1_score, "scores": scores, "y": y}

    def predict_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, verbose=True
        )
        return Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {state_dict[k].shape}"
                    )
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.val_outputs, dim=0).reshape(-1)

        all_labels = torch.cat(self.val_y, dim=0)

        loss = self.loss_fn(all_preds.reshape(-1), all_labels.reshape(-1).float())

        pred_labels = (torch.sigmoid(all_preds).reshape(-1) > 0.9).long()
        accuracy = self.accuracy(pred_labels, all_labels)
        f1_score = self.f1_score(pred_labels, all_labels)
        recall = self.recall(pred_labels, all_labels)
        specificity = self.specificity(pred_labels, all_labels)
        auc = self.roc_auc(all_preds, all_labels)
        precision = self.precision(pred_labels, all_labels)

        self.log_dict(
            {
                "val_loss_end": loss,
                "val_accuracy_end": accuracy,
                "val_f1_score_end": f1_score,
                "val_recall_end": recall,
                "val_specificity_end": specificity,
                "val_auc_end": auc,
                "val_precision_end": precision,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.val_outputs.clear()
        self.val_y.clear()
        return {"val_loss": loss, "val_accuracy": accuracy, "val_f1_score": f1_score}

    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_outputs, dim=0).reshape(-1)

        all_labels = torch.cat(self.test_y, dim=0)

        loss = self.loss_fn(all_preds.reshape(-1), all_labels.reshape(-1).float())

        pred_labels = (torch.sigmoid(all_preds).reshape(-1) > 0.5).long()
        accuracy = self.accuracy(pred_labels, all_labels)
        f1_score = self.f1_score(pred_labels, all_labels)
        recall = self.recall(pred_labels, all_labels)
        specificity = self.specificity(pred_labels, all_labels)
        auc = self.roc_auc(all_preds, all_labels)
        precision = self.precision(pred_labels, all_labels)

        self.log_dict(
            {
                "test_loss_end": loss,
                "test_accuracy_end": accuracy,
                "test_f1_score_end": f1_score,
                "test_recall_end": recall,
                "test_specificity_end": specificity,
                "test_auc_end": auc,
                "test_precision_end": precision,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.val_outputs.clear()
        self.val_y.clear()
        return {"val_loss": loss, "val_accuracy": accuracy, "val_f1_score": f1_score}


def test_model():
    # Define model parameters
    input_size = 1 
    output_size = 1
    d_model = 16
    max_len = 100
    learning_rate = 1e-4
    weight_decay = 0
    padding_idx = 0.0

    # Create model
    model = ConvModel(
        input_size=input_size,
        output_size=output_size,
        d_model=d_model,
        max_len=max_len,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        padding_idx=padding_idx,
    )

    # Create random input
    x = torch.randn(2, 23, 3)
    y = torch.tensor([1])
    
    # Forward pass
    output = model(x)
    print(output.shape)

if __name__ == "__main__":
    test_model()
