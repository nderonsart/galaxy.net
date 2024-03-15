import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class CNN(nn.Module):
    """
    The Convolutional Neural Network model
    """

    def __init__(self):
        """
        Initialize the model
        The model architecture is:
        - 3 convolutional layers
        - 1 pooling layer
        - 2 fully connected linear layers
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 37 * 37, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """
        The forward pass of the model
        parameters:
            - x: input tensor
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.reshape(-1, 64 * 37 * 37)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(model, X_train, y_train, X_test, y_test,
          epochs=10, batch_size=32, lr=0.001):
    """
    Train the model
    parameters:
        - model: the model to train
        - X_train: training images
        - y_train: training labels
        - X_test: testing images
        - y_test: testing labels
        - epochs: number of epochs
        - batch_size: batch size
        - lr: learning rate
    """
    X_train = torch.from_numpy(X_train).float().permute(0, 3, 1, 2)
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float().permute(0, 3, 1, 2)
    y_test = torch.from_numpy(y_test).float()

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_dl:
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)

        model.eval()
        test_loss = 0.0
        y_pred = []
        for X, y in test_dl:
            with torch.no_grad():
                output = torch.sigmoid(model(X)).squeeze(1)
                loss = criterion(output, y)
                test_loss += loss.item() * X.size(0)
        test_loss /= len(test_dl.dataset)

        print(f'Epoch: {epoch + 1}/{epochs}', end='')
        print(f'| Train loss: {train_loss:.4f}', end='')
        print(f'| Test loss: {test_loss:.4f}')
