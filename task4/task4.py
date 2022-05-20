import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/task4

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
            # nn.Linear(in_features=128, out_features=128),
            # nn.PReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=64),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64, out_features=1),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Linear(32, 1)
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


def train(model, epochs, train_loader, optimizer, scheduler, mode, to_save = False):
    outputs = []
    losses = []
    pbar = tqdm(desc='Training', total= epochs * len(train_loader), leave=False)
    for epoch in range(epochs):
        print(f"\n -- Epoch {epoch + 1} / {epochs} --")
        if mode == "AE":
            for i, x in enumerate(train_loader):
                x_ = x["inp"].float().to(device)
                reconstructed = model(x_)
                loss = loss_function(reconstructed, x_)
                pbar.set_description_str(f'Epoch {epoch + 1}, batch {i + 1}')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss)
                pbar.set_postfix_str(f' loss: {loss:.3e}')
                pbar.update()
            outputs.append((epoch, x_, reconstructed))
            # print(outputs)
        elif mode == "pretrain":
            for i, batch in enumerate(train_loader):
                pbar.set_description_str(f'Epoch {epoch + 1}, batch {i + 1}')
                x_train, y_train = batch['inp'].to(device), batch['oup'].to(device)
                output = model(x_train.float())
                loss = loss_function(output, y_train.float())
                pbar.set_description_str(f'Epoch {epoch + 1}, batch {i + 1}')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss)
                pbar.set_postfix_str(f' loss: {loss:.3e}')
                pbar.update()
            outputs.append((epoch, y_train.float(), output))
            # print(outputs)
        pbar.refresh()
        if scheduler is not None:
            scheduler.step()
    return outputs


BATCH_SIZE = 256

if __name__ == '__main__':
    path = os.getcwd()
    dpath = os.path.join(path, 'data')
    data = Data(dpath)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_function = nn.MSELoss()

    # Training autoencoder
    model_AE = AutoEncoder().to(device)
    optimizer = torch.optim.Adam(model_AE.parameters(),
                                lr=1e-1,
                                weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    ptrain_set = MoleculeDataSet(data,
                                Data_reduced=[],
                                mode="AE")
    ptrain_loader = DataLoader(ptrain_set, batch_size=BATCH_SIZE,
                            shuffle=True,
                            drop_last=False,
                            )

    outputs = train(model_AE, 100, ptrain_loader, optimizer, scheduler, mode="AE")


    # get the reduced features
    ptrain_tensor = torch.from_numpy(data.pfeatures).float().to(device)
    pfeatures_reduced = model_AE.encoder(ptrain_tensor).detach().cpu().numpy()

    # pre-train 50000 data with labels
    model_nn = NeuralNetwork().to(device)
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
                    optimizer, scheduler, mode="pretrain")
