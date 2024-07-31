import torch
import pytorch_lightning as pl
import joblib
import numpy as np
from datamodule import TractogramDM
from tqdm import tqdm
from net import ConvModel  # Assuming ConvModel is defined in model.py
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from intersection import CustomLinearRegression
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


def get_predictions(model, datamodule, noise=False, times=10):
    #trainer = pl.Trainer(accelerator="gpu", devices=1)

    # transfer model to GPU
    model = model.to("cuda")
    predictions = []
    for batch in tqdm(datamodule):
        batch = batch[0].to("cuda")
        if noise:
            model.model.train()
            preds=0
            # add dropout noise to batch
            for i in range(times):
                preds += torch.sigmoid(model.model(batch)).detach().cpu()
            predictions.append(preds/times)

        else:
            model.model.eval()

            

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
    
    # Load data modules for test set
    sift_dm = TractogramDM(data_dir, label_dir, batch_size=batch_size, num_workers=6, mode="sift", binarize=False)
    commit_dm = TractogramDM(data_dir, label_dir, batch_size=batch_size, num_workers=6, mode="commit", binarize=False)
    sift_dm.setup(stage='test')
    commit_dm.setup(stage='test')
    sift_test_dataloader = sift_dm.test_dataloader()
    commit_test_dataloader = commit_dm.test_dataloader()

    # Load models
    sift_model = load_model("checkpoints/sift_val-epoch=02-val_loss=0.32.ckpt", input_size, num_labels, learning_rate, weight_decay)
    commit_model = load_model("checkpoints/commit_train-epoch=02-train_loss=0.59.ckpt", input_size, num_labels, learning_rate, weight_decay)

    # Get predictions
    sift_predictions = get_predictions(sift_model, sift_test_dataloader, noise=False).reshape(-1, 1)
    commit_predictions = get_predictions(commit_model, commit_test_dataloader, noise=False).reshape(-1, 1)

    # Load the regression model
    reg = joblib.load('linear_regression_model.pkl')

    # Combine predictions
    X_test = np.hstack((sift_predictions, commit_predictions))
 #   print(X_test.shape)
    combined_predictions = reg.predict(X_test)
 #   combined_predictions = (sift_predictions + commit_predictions) / 2

    # check recall, specificity, precision, f1 score, auc

        # Load test labels
    test_labels = []
    np_labels = []
    for sift_batch, commit_batch in zip(sift_test_dataloader, commit_test_dataloader):
        test_labels.append((sift_batch[1]>=1) * (commit_batch[1]>=1))
        np_labels.append(np.array([sift_batch[1], commit_batch[1]]).T.reshape(-1, 2))
    
    test_labels = np.concatenate(test_labels, axis=0)


    np_labels = np.concatenate(np_labels, axis=0)
    # find indices in np_labels where both are 0 or 1
    p_indices = np.where(np.logical_and(np_labels[:,0] >= 1, np_labels[:, 1] >= 1))[0]
    n_indices = np.where(np.logical_and(np_labels[:, 0] < 1e-6, np_labels[:, 1] <1e-6))[0]
    print(len(p_indices))
    print(len(n_indices))
    # concatenate the indices
    indices = np.concatenate((p_indices, n_indices), axis=0)
    # i indices are all but the ones in indices
    i_indices = np.setdiff1d(np.arange(len(np_labels)), indices)
    
    # N/P
  #  indices = indices

    # I/P 
  #  indices = np.concatenate((i_indices, p_indices), axis=0)

  #  np_labels = test_labels[indices]
    # I/N
    indices = np.concatenate((i_indices, n_indices), axis=0)
    # make I labels 1 and N labels 0
    labels = np.zeros(len(test_labels))
    labels[i_indices] = 1
    np_labels = labels[indices]
    print(np_labels.shape)

    # get the labels
    np_combined_predictions = combined_predictions[indices]

    print(np_combined_predictions.shape)
    # Compute evaluation metrics
    auc_roc = roc_auc_score(test_labels, combined_predictions)
    combined_predictions = combined_predictions > 0.008
    accuracy = accuracy_score(test_labels, combined_predictions)
    tn, fp, fn, tp = confusion_matrix(test_labels, combined_predictions).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    print(f"AUC-ROC: {auc_roc}")
    print(f"Accuracy: {accuracy}")
    print(f"Specificity: {specificity}")
    print(f"Sensitivity: {sensitivity}")


    # same with np labels
    auc_roc = roc_auc_score(np_labels, np_combined_predictions)
    np_combined_predictions = np_combined_predictions > 0.005
    accuracy = accuracy_score(np_labels, np_combined_predictions)
    tn, fp, fn, tp = confusion_matrix(np_labels, np_combined_predictions).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    print(f"NP AUC-ROC: {auc_roc}")
    print(f"NP Accuracy: {accuracy}")
    print(f"NP Specificity: {specificity}")
    print(f"NP Sensitivity: {sensitivity}")


if __name__ == "__main__":
    main()
