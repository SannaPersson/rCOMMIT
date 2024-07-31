import torch
import pytorch_lightning as pl
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from tqdm import tqdm

from datamodule import TractogramDM
from net import ConvModel  # Assuming ConvModel is defined in model.py
torch.set_float32_matmul_precision("medium")

class CustomLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, standard_value=0.5):
        self.alpha = alpha
        self.standard_value = standard_value
        self.model = LinearRegression()

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.model.fit(X, y)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        return self

    def predict(self, X):
        check_is_fitted(self, ['coef_', 'intercept_'])
        X = check_array(X)
        return self.model.predict(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return -self.custom_loss(y, y_pred)

    def custom_loss(self, y_true, y_pred):
        # Standard value penalty
        standard_penalty = np.sum((self.coef_ - self.standard_value) ** 2)
        # Regular MSE loss
        mse_loss = np.mean((y_true - y_pred) ** 2)
        # Total loss
        total_loss = mse_loss# + self.alpha * standard_penalty
        return total_loss

def load_model(checkpoint_path, input_size, num_labels, learning_rate, weight_decay):
    model = ConvModel.load_from_checkpoint(
        checkpoint_path,
        strict=False,
        input_size=input_size,
        output_size=num_labels,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    model.eval()
    return model

def get_predictions(model, datamodule):
    #trainer = pl.Trainer(accelerator="gpu", devices=1)
    # transfer model to GPU
    model = model.to("cuda")
    predictions = []
    for batch in tqdm(datamodule):
        batch = batch[0].to("cuda")
        predictions.append(torch.sigmoid(model.model(batch)).detach().cpu())
    return torch.cat(predictions, dim=0).cpu().numpy()

def main():
    # Hyperparameters
    input_size = 3
    num_labels = 1
    learning_rate = 1e-4
    weight_decay = 5e-5
    batch_size = 128
    data_dir = "../../Data/TRK_chunks"
    label_dir = "../../Data/Intersection_Labels"
    
    # Load data modules for validation set
    sift_dm = TractogramDM(data_dir, label_dir, batch_size=batch_size, num_workers=6, mode="sift", binarize=False)
    commit_dm = TractogramDM(data_dir, label_dir, batch_size=batch_size, num_workers=6, mode="commit", binarize=False)
    sift_dm.setup(stage='validate')
    commit_dm.setup(stage='validate')
    sift_val_dataloader = sift_dm.val_dataloader()
    commit_val_dataloader = commit_dm.val_dataloader()

    # Load models
    sift_model = load_model("checkpoints/sift_val-epoch=02-val_loss=0.32.ckpt", input_size, num_labels, learning_rate, weight_decay)
    commit_model = load_model("checkpoints/commit_train-epoch=02-train_loss=0.59.ckpt", input_size, num_labels, learning_rate, weight_decay)

    # Get predictions
    sift_predictions = get_predictions(sift_model, sift_val_dataloader)
    commit_predictions = get_predictions(commit_model, commit_val_dataloader)
    print(sift_predictions.mean(), commit_predictions.mean())
    # Load labels and compute intersection
    sift_labels = []
    commit_labels = []
    for sift_batch, commit_batch in zip(sift_val_dataloader, commit_val_dataloader):
        sift_labels.append(sift_batch[1].cpu().numpy())
        commit_labels.append(commit_batch[1].cpu().numpy())
    sift_labels = np.concatenate(sift_labels, axis=0)
    commit_labels = np.concatenate(commit_labels, axis=0)
    intersection_labels = np.logical_and(sift_labels>=1, commit_labels>=1).astype(int)
    
    # Check negative/implausible labels
    # check  indices when labels are < 1
    sift_indices = np.where(sift_labels < 1)
    commit_indices = np.where(commit_labels < 1)
    # union of the two sets
    union_indices = np.union1d(sift_indices, commit_indices)
    # keep only union indices
    sift_predictions = sift_predictions[union_indices]
    commit_predictions = commit_predictions[union_indices]
    intersection_labels = np.ones(len(intersection_labels))
    # check where sift and commit are 0
    sift_indices = np.where(sift_labels == 0)
    commit_indices = np.where(commit_labels == 0)
    # intersection of the two sets
    intersection_indices = np.intersect1d(sift_indices, commit_indices)
    # set intersection labels to 0
    intersection_labels[intersection_indices] = 0
    intersection_labels = intersection_labels[union_indices]

    
    # Train linear regression model
    X = np.concatenate((sift_predictions.reshape(-1, 1), commit_predictions.reshape(-1, 1)), axis=1)
    print(X.mean(0))
    y = intersection_labels
 #   reg = LinearRegression().fit(X, y)    # Save the regression model
    # as above but weight the labels by the number of times they appear in the dataset
    weight = np.sum(y) / y.shape[0]
    class_weight = {0: 1, 1: 1/weight}
    reg = CustomLinearRegression(alpha=10).fit(X, y)
    # print the coefficients
    joblib.dump(reg, 'linear_regression_model_ni.pkl')
    y = (y>=1).astype(np.int32)
    # Evaluate the combined model
    combined_predictions = reg.predict(X)
    # calculate auc
    # import roc_auc_score
    auc = roc_auc_score(y, combined_predictions)

    combined_predictions = (combined_predictions > 0.01).astype(int)  # Assuming binary classification
    accuracy = accuracy_score(y, combined_predictions)
    confusion = confusion_matrix(y, combined_predictions)
    # sensitivity
    sensitivity = confusion[1,1] / (confusion[1,0] + confusion[1,1])
    # specificity
    specificity = confusion[0,0] / (confusion[0,0] + confusion[0,1])
    mse = np.mean((y - combined_predictions) ** 2)
    print(f"Combined model accuracy: {accuracy}")
    print(f"Combined model AUC: {auc}")
    print(f"Combined model sensitivity: {sensitivity}")
    print(f"Combined model specificity: {specificity}")
    print(f"Combined model MSE: {mse}")
if __name__ == "__main__":
    main()
