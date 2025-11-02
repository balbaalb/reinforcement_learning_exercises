import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
import pandas as pd


class GameNetworkModel(nn.Module):
    def __init__(self):
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
        xt = self.scale_torchize_x(x)
        yt = torch.FloatTensor(y).reshape(-1, 1)
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(xt)
            loss = self.criterion(y_pred, yt)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % (self.epochs // 100) == 0:
                print(f"Epoch: {epoch + 1}, loss: {loss.item()}")

    def predict(self, x: npt.ArrayLike) -> npt.ArrayLike:
        xt = self.scale_torchize_x(x)
        y_pred = self.model(xt)
        return y_pred.detach().numpy()
