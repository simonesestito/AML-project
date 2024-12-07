import torch.nn as nn
import lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from ..llama import LlamaInstruct
from ..classifier import LightningHiddenStateSAPLMA
from ..dataset import StatementDataModule


def try_to_overfit(llama: LlamaInstruct,saplma_version: nn.Module,reduction: nn.Module, datamodule: StatementDataModule, batch_size: int = 64, lr:float = 1e-5):
    """
    Try to overfit the model on a single batch
    """
    model = LightningHiddenStateSAPLMA(llama, saplma_version, reduction, lr=lr)
    model.hparams.batch_size = batch_size
    trainer = pl.Trainer(overfit_batches=1, max_epochs=100, log_every_n_steps=1)
    trainer.fit(model=model, train_dataloaders=datamodule.train_dataloader(shuffle=False))

def plot_weight_matrix(weight_matrix, x_label, y_label):
    """
    Plot a weight matrix as a heatmap
    """
    assert len(weight_matrix.shape)==2 or len(weight_matrix.shape)==1, "Weight matrix should be one or bi-dimensional"
    w,h = 20,8
    if len(weight_matrix.shape)==1:
        weight_matrix = weight_matrix.unsqueeze(0)
        h=2
    plt.figure(figsize=(w, h))
    sns.heatmap(
        weight_matrix,
        cmap="viridis",  # Use the viridis colormap
        annot=False,     # Set to True if you want to annotate values
        cbar=True        # Display the color bar
    )

    # Add labels and a title
    plt.title("Weight Matrix Visualization")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()