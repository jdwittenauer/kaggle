import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

class TorchNet(nn.Module):
    def __init__(self, input_size):
        super(TorchNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = TorchNet(X.shape[1])
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print(model)

def create_dataloaders(X, y, batch_size, val_data=None):
    train_ds = TensorDataset(torch.Tensor(X.values), torch.Tensor(y.values))
    train_dl = DataLoader(train_ds, batch_size)
    if val_data is not None:
        X_val, y_val = val_data
        val_ds = TensorDataset(torch.Tensor(X_val.values), torch.Tensor(y_val.values))
        val_dl = DataLoader(val_ds, batch_size)
        return train_dl, val_dl
    else:
        return train_dl

train_dl, val_dl = create_dataloaders(X, y, batch_size, val_data=(X_val, y_val))

def fit(model, optimizer, loss_fn, train_dl, n_epochs, val_dl=None):
    for epoch in range(n_epochs):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_val_loss = 0.0
        steps = 0
        val_steps = 0
        for i, data in enumerate(train_dl, 0):
            X, y = data
            y = y.view(-1, 1)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1
        if val_dl is not None:
            model.eval()
            for i, data in enumerate(val_dl, 0):
                X, y = data
                y = y.view(-1, 1)
                y_pred = model(X)
                val_loss = loss_fn(y_pred, y)
                epoch_val_loss += val_loss.item()
                val_steps += 1
            t1 = time.time()
            print('[Epoch {0:d}] loss: {1:.3f} | val loss: {2:.3f} | {3:.0f} s'.format(
                epoch + 1, epoch_loss / steps, epoch_val_loss / val_steps, t1 - t0))
        else:
            t1 = time.time()
            print('[Epoch {0:d}] loss: {1:.3f} | {2:.0f} s'.format(epoch + 1, epoch_loss / steps, t1 - t0))

fit(model, optimizer, loss_fn, train_dl, n_epochs=n_epochs, val_dl=val_dl)
