import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/task4
os.getcwd()


class Data:
    def __init__(self, dpath):
        self.pfeatures = pd.read_csv(os.path.join(
            dpath, 'pretrain_features.csv')).to_numpy()[:, 2:].astype(float)
        self.plabels = pd.read_csv(os.path.join(
            dpath, 'pretrain_labels.csv')).to_numpy()[:, 2:].astype(float)
        self.tfeatures = pd.read_csv(os.path.join(
            dpath, 'train_features.csv')).to_numpy()[:, 2:].astype(float)
        self.tlabels = pd.read_csv(os.path.join(
            dpath, 'train_labels.csv')).to_numpy()[:, 2:].astype(float)
        self.testfeatures = pd.read_csv(os.path.join(
            dpath, 'test_features.csv')).to_numpy()[:, 2:].astype(float)


path = os.getcwd()
dpath = os.path.join(path, 'data')
data = Data(dpath)


class AutoEncoder(nn.Module):
    def __init__(self):
        self.Encoded_dim = 128
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=1000, out_features=512),
            nn.PReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=self.Encoded_dim),
            # nn.PReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.Encoded_dim, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.PReLU(),
            nn.Linear(in_features=512, out_features=1000),
        )

    def forward(self, x):
        enc = self.encoder(x)
        return self.decoder(enc)


class MoleculeDataSet(Dataset):
    def __init__(self, Data):
        self.data = Data

    def __getitem__(self, index):
        a = self.data[index, :]
        return a

    def __len__(self):
        return self.data.shape[0]


def train(model, epochs, train_loader):
    outputs = []
    losses = []
    for epoch in range(epochs):
        for i, x in enumerate(train_loader):
            reconstructed = model(x.float())
            loss = loss_function(reconstructed, x.float())
            print("Batch ", i, ": ", loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss)
            outputs.append((epochs, x.float(), reconstructed))
    return outputs


BATCH_SIZE = 100
model = AutoEncoder()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-1,
                             weight_decay=1e-8)

train_set = MoleculeDataSet(data.pfeatures)
train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

outputs = train(model, 20, train_loader)
