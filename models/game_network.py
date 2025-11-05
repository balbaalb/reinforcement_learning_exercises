import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import numpy.typing as npt
import matplotlib.pylab as plt

"""
Using pytorch, a game model is built. 
"""


class GameNetworkModel(nn.Module):
    """
    A simple 3-layer fully feed formward neural network (MLP)
    """

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=2, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
        )

    def forward(self, x):
        return self.network(x)


class GameNetwork:
    """
    Functionalities for fitting (training) the GameNetworkModel object and then making predictions with it.
    """

    def __init__(self, size: int, epochs: int, lr: int):
        torch.manual_seed(42)
        self.size = size
        self.epochs = epochs
        self.lr = lr
        self.model = GameNetworkModel()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def scale_torchize_x(self, x) -> torch.FloatTensor:
        xt = np.zeros_like(x)
        xt[:, 0] = x[:, 0] / (self.size * self.size - 1.0)
        xt[:, 1] = x[:, 1] / 3.0
        xt = torch.FloatTensor(x)
        return xt

    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike) -> None:
        train_size = int(len(x) * 0.8)
        val_size = len(x) - train_size
        xt = self.scale_torchize_x(x)
        yt = torch.FloatTensor(y).reshape(-1, 1)
        dataset = TensorDataset(xt, yt)
        dataset_train, dataset_val = random_split(
            dataset=dataset, lengths=[train_size, val_size]
        )
        dataloader_train = DataLoader(
            dataset=dataset_train, batch_size=100, shuffle=True
        )
        dataloader_val = DataLoader(dataset=dataset_val, batch_size=100, shuffle=True)
        losses_train = []
        losses_val = []
        for epoch in range(self.epochs):
            epoch_loss = 0
            for x_batch, y_batch in dataloader_train:
                self.optimizer.zero_grad()
                y_pred = self.model(x_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            losses_train.append(epoch_loss / len(dataloader_train))
            epoch_val_loss = 0
            with torch.no_grad():
                for x_batch, y_batch in dataloader_val:
                    y_pred = self.model(x_batch)
                    loss = self.criterion(y_pred, y_batch)
                    epoch_val_loss += loss.item()
            losses_val.append(epoch_val_loss / len(dataloader_val))
            if (epoch + 1) % (self.epochs // 100) == 0:
                print(
                    f"Epoch: {epoch + 1}, loss_training: {losses_train[-1]}, loss_val = {losses_val[-1]}"
                )
        plt.plot(losses_train, label="Training Loss")
        plt.plot(losses_val, label="Validation Loss")
        plt.yscale("log")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def predict(self, x: npt.ArrayLike) -> npt.ArrayLike:
        xt = self.scale_torchize_x(x)
        y_pred = self.model(xt)
        return y_pred.detach().numpy()
