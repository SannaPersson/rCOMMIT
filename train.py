from net import ConvModel
from datamodule import TractogramDM
from callbacks import CheckpointEveryNSteps
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

torch.set_float32_matmul_precision("medium")

def train(
    mode,
    input_size,
    num_classes,
    num_labels,
    learning_rate,
    weight_decay,
    grad_norm,
    batch_size,
    num_epochs,
    data_dir,
    label_dir,
    save_freq,
    resume_from_checkpoint,
    resume,
    overfit_batches,
    logger,
    checkpoint_callback_train,
    checkpoint_callback_val,
    test=False,
):
    if resume:
        model = ConvModel.load_from_checkpoint(
            resume_from_checkpoint,
            strict=False,
            input_size=input_size,
            output_size=num_labels,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        model = ConvModel(
            input_size=input_size,
            d_model=16,
            output_size=num_labels,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    dm = TractogramDM(
        data_dir, label_dir, batch_size=batch_size, num_workers=6, mode=mode
    )
    # loop through and print every parameter in the model
    for name, param in model.named_parameters():
        print(name, param.shape)


    if overfit_batches:
        trainer = pl.Trainer(
            logger=logger,
            val_check_interval=0.07,
            # limit_val_batches=,
            check_val_every_n_epoch=3000,
            accelerator="gpu",
            devices=1,
            min_epochs=1,
            max_epochs=500,
            precision="16-mixed",
            callbacks=[checkpoint_callback_train, checkpoint_callback_val],
            # gradient_clip_val=grad_norm,
            # gradient_clip_algorithm="norm",
            overfit_batches=1,
            deterministic=True,
        )
    else:
        trainer = pl.Trainer(
            logger=logger,
            val_check_interval=0.07,
            limit_val_batches=0.3,
            accelerator="gpu",
            devices=1,
            min_epochs=1,
            max_epochs=num_epochs,
            precision="16-mixed",
            callbacks=[checkpoint_callback_train, checkpoint_callback_val],
        )
    if test:
        trainer.test(model, dm)
    else:
        trainer.fit(model, dm)
        trainer.validate(model, dm)


if __name__ == "__main__":
    # Set device cuda for GPU if it's available otherwise run on the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    mode = "sift"
    input_size = 3 
    num_classes = 2
    num_labels = 1
    learning_rate = 1e-4
    weight_decay = 5e-5
    grad_norm = 3.0
    batch_size = 128 
    num_epochs = 3 
    data_dir = "../../Data/TRK_chunks"
    label_dir = "../../Data/Intersection_Labels"
    save_freq = 10000
    overfit_batches = False
    logger_name = f"{mode}_model"
    resume_from_checkpoint = "r-sift_weights.ckpt"
    resume = True 
    checkpoint_callback_train = ModelCheckpoint(
        dirpath="checkpoints",
        filename=mode + "_train-{epoch:02d}-{train_loss:.2f}",
        verbose=True,
        monitor="train_loss",
        mode="min",
        save_last=True,
        every_n_train_steps=save_freq,
    )
    checkpoint_callback_val = ModelCheckpoint(
        dirpath="checkpoints",
        filename=mode + "_val-{epoch:02d}-{val_loss:.2f}",
        verbose=True,
        monitor="val_loss",
        mode="min",
        save_last=False,
        every_n_train_steps=5000,
    )
    logger = TensorBoardLogger("tb_logs", name=logger_name)
    test =True 
    train(
        mode,
        input_size,
        num_classes,
        num_labels,
        learning_rate,
        weight_decay,
        grad_norm,
        batch_size,
        num_epochs,
        data_dir,
        label_dir,
        save_freq,
        resume_from_checkpoint,
        resume,
        overfit_batches,
        logger,
        checkpoint_callback_train,
        checkpoint_callback_val,
        test,
    )

