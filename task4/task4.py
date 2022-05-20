import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import random

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/task4
os.getcwd()


class Data:
    def __init__(self, dpath):
        self.pfeatures = pd.read_csv(os.path.join(
            dpath, 'pretrain_features.csv')).to_numpy()[:, 2:].astype(float)
        self.plabels = pd.read_csv(os.path.join(
            dpath, 'pretrain_labels.csv')).to_numpy()[:, 1:].astype(float)
        self.tfeatures = pd.read_csv(os.path.join(
            dpath, 'train_features.csv')).to_numpy()[:, 2:].astype(float)
        self.tlabels = pd.read_csv(os.path.join(
            dpath, 'train_labels.csv')).to_numpy()[:, 1:].astype(float)
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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1),
        )

    def forward(self, x):
        outputs = self.linear_relu_stack(x)
        return outputs


class MoleculeDataSet(Dataset):
    def __init__(self, Data, Data_reduced, mode):
        self.mode = mode
        if self.mode == 'AE':
            self.inp = Data.pfeatures
        elif self.mode == 'pretrain':
            self.inp = Data_reduced
            self.oup = Data.plabels

    def __getitem__(self, index):
        if self.mode == 'AE':
            inpt = self.inp[index, :]
            return {'inp': inpt
                    }

        elif self.mode == 'pretrain':
            inpt = self.inp[index, :]
            oupt = self.oup[index]
            return {'inp': inpt,
                    'oup': oupt,
                    }

    def __len__(self):
        return self.inp.shape[0]


def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)


def train(model, epochs, train_loader, optimizer, mode):
    outputs = []
    losses = []
    for epoch in range(epochs):
        if mode == "AE":
            for i, x in enumerate(train_loader):
                reconstructed = model(x["inp"].float())
                loss = loss_function(reconstructed, x["inp"].float())
                print("Batch ", i, ": ", loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss)
            outputs.append((epoch, x["inp"].float(), reconstructed))
            print(outputs)
        elif mode == "pretrain":
            for i, batch in enumerate(train_loader):
                x_train, y_train = batch['inp'], batch['oup']
                output = model(x_train.float())
                loss = loss_function(output, y_train.float())
                print("Batch ", i, ": ", loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss)
            outputs.append((epoch, y_train.float(), output))
            print(outputs)
    return outputs


loss_function = torch.nn.MSELoss()
BATCH_SIZE = 100

# Training autoencoder
model_AE = AutoEncoder()
optimizer = torch.optim.Adam(model_AE.parameters(),
                             lr=1e-1,
                             weight_decay=1e-8)

ptrain_set = MoleculeDataSet(data,
                             Data_reduced=[],
                             mode="AE")
ptrain_loader = DataLoader(ptrain_set, batch_size=BATCH_SIZE,
                           shuffle=True,
                           drop_last=False,
                           )

outputs = train(model_AE, 100, ptrain_loader, optimizer, mode="AE")


# get the reduced features
ptrain_tensor = torch.from_numpy(data.pfeatures).float()
pfeatures_reduced = model_AE.encoder(ptrain_tensor).detach().numpy()

# pre-train 50000 data with labels
model_nn = NeuralNetwork()
optimizer = torch.optim.Adam(model_nn.parameters(),
                             lr=1e-3
                             )

ptrain_reduced_set = MoleculeDataSet(data,
                                     Data_reduced=pfeatures_reduced,
                                     mode="pretrain")
ptrain_reduced_loader = DataLoader(ptrain_reduced_set, batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   drop_last=False,
                                   )

outputs = train(model_nn, 100, ptrain_reduced_loader,
                optimizer, mode="pretrain")
