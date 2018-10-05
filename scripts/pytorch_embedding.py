import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

class TorchEmbeddingNet(nn.Module):
    def __init__(self, cat_vars, cont_vars, embedding_sizes):
        super(TorchNet, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(c, s) for c, s in embedding_sizes])

        self.n_cat = len(cat_vars)
        self.n_cont = len(cont_vars)
        self.n_embed = sum(e.embedding_dim for e in self.embeddings)
        
        self.fc1 = nn.Linear(self.n_embed + self.n_cont, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.out = nn.Linear(500, 1)
        
        self.bn_cont = nn.BatchNorm1d(self.n_cont)
        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(500)
        
        self.d_embed = nn.Dropout(0.04)
        self.d1 = nn.Dropout(0.001)
        self.d2 = nn.Dropout(0.01)
        
        for e in self.embeddings:
            e = e.weight.data
            sc = 2 / (e.size(1) + 1)
            e.uniform_(-sc, sc)

        nn.init.kaiming_normal(self.fc1.weight.data)
        nn.init.kaiming_normal(self.fc2.weight.data)
        nn.init.kaiming_normal(self.out.weight.data)

    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.d_embed(x)
        
        x2 = self.bn_cont(x_cont)
        x = torch.cat([x, x2], 1)
        
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.d1(x)
        
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.d2(x)
        
        x = self.out(x)

        return x

model = TorchEmbeddingNet(cat_vars, cont_vars, embedding_sizes)
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(model)

def create_dataloaders(X, y, batch_size, val_data=None):
    X_cat = X[cat_vars].values.astype('int64')
    X_cont = X[cont_vars].values.astype('float32')
    y = y.values.astype('float32')
    train_ds = TensorDataset(torch.from_numpy(X_cat), torch.from_numpy(X_cont), torch.from_numpy(y))
    train_dl = DataLoader(train_ds, batch_size)
    if val_data is not None:
        X_val, y_val = val_data
        X_val_cat = X_val[cat_vars].values.astype('int64')
        X_val_cont = X_val[cont_vars].values.astype('float32')
        y_val = y_val.values.astype('float32')
        val_ds = TensorDataset(torch.from_numpy(X_val_cat), torch.from_numpy(X_val_cont), torch.from_numpy(y_val))
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
            X_cat, X_cont, y = data
            y = y.view(-1, 1)
            y_pred = model(X_cat, X_cont)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1
        if val_dl is not None:
            model.eval()
            for i, data in enumerate(val_dl, 0):
                X_cat, X_cont, y = data
                y = y.view(-1, 1)
                y_pred = model(X_cat, X_cont)
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
