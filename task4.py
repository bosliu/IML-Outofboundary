import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from dataclasses import dataclass
from copy import deepcopy
import random
import csv


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
        self.test_ID = pd.read_csv(os.path.join(
            dpath, 'test_features.csv')).to_numpy()[:, 0]

        self.all_features = np.vstack(
            (self.pfeatures, self.tfeatures, self.testfeatures))


class AutoEncoder(nn.Module):
    def __init__(self):
        self.Encoded_dim = ENCODING_DIM
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=1000, out_features=512),
            nn.PReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=self.Encoded_dim),
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
        # self.linear_relu_stack = nn.Sequential(
        #     # nn.Linear(in_features=128, out_features=128),
        #     # nn.PReLU(),
        #     # nn.Dropout(p=0.2),
        #     nn.Linear(in_features=128, out_features=64),
        #     nn.PReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=64, out_features=32),
        #     nn.PReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(32, 1)
        # )
        self.layer1 = nn.Sequential(
            nn.Linear(ENCODING_DIM, 128),
            nn.PReLU(),
        )
        self.layer2 = nn.Sequential(
            # nn.Dropout(p=0.2),
            # nn.Linear(128, 64),
            # nn.PReLU(),
            nn.Identity(),  # placeholder
        )
        self.layer3 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.PReLU(),
        )
        self.layer4 = nn.Sequential(
            # nn.Dropout(p=0.2),
            nn.Linear(64, 1),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


class MoleculeDataSet(Dataset):
    def __init__(self, Data, Data_reduced, mode):
        self.mode = mode
        if self.mode == 'AE':
            self.inp = Data.all_features
        elif self.mode == 'pretrain':
            self.inp = Data_reduced
            self.oup = Data.plabels
        elif self.mode == 'retrain':
            self.inp = Data_reduced
            self.oup = Data.tlabels

    def __getitem__(self, index):
        if self.mode == 'AE':
            inpt = self.inp[index, :]
            return {'inp': inpt
                    }

        elif self.mode == 'pretrain' or self.mode == 'retrain':
            inpt = self.inp[index, :]
            oupt = self.oup[index]
            return {'inp': inpt,
                    'oup': oupt,
                    }

    def __len__(self):
        return self.inp.shape[0]


@dataclass
class AverageMeter:
    name: str = ""
    fmt: str = ':f'
    val: float = 0
    avg: float = 0
    sum: float = 0
    count: int = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f'{self.name}: {self.val:.3e}, avg: {self.avg:.3e}'


# def worker_init_fn(worker_id):
#     torch_seed = torch.initial_seed()
#     random.seed(torch_seed + worker_id)
#     if torch_seed >= 2**30:
#         torch_seed = torch_seed % 2**30
#     np.random.seed(torch_seed + worker_id)


def train(model, epochs, train_loader, validation_loader, optimizer, scheduler, mode, to_save=False):
    outputs = []
    pbar = tqdm(desc='Training', total=epochs * len(train_loader), leave=False)
    losses_val_hist = 1e3 if validation_loader else None
    val_loss_inc_epochs, val_loss_inc_epochs_max = 0, 10
    model_best = None
    for epoch in range(epochs):
        model.train()
        losses = AverageMeter('Loss', ':.3e')
        if mode == "AE":
            for x in train_loader:
                x_ = x["inp"].float().to(device)
                reconstructed = model(x_)
                loss = loss_function(reconstructed, x_)
                pbar.set_description_str(f'Epoch {epoch + 1}')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.update(loss.item(), x_.shape[0])
                pbar.set_postfix_str("{}".format(losses))
                pbar.update()
            outputs.append((epoch, x_, reconstructed))
            # print(outputs)
        elif mode == "train":
            for i, batch in enumerate(train_loader):
                pbar.set_description_str(f'Epoch {epoch + 1}')
                x_train, y_train = batch['inp'].to(
                    device), batch['oup'].to(device)
                output = model(x_train.float())
                loss = loss_function(output, y_train.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.update(loss.item(), x_train.shape[0])
                pbar.set_postfix_str("{}".format(losses))
                pbar.update()
            outputs.append((epoch, y_train.float(), output))
            # print(outputs)
        if loss.item() > 1e2:
            raise Exception("‼ Abnormal loss ‼")
        # validation
        pbar.refresh()
        if validation_loader:
            model.eval()
            losses_val = AverageMeter('Loss_val', ':.3e')
            for x in validation_loader:
                if mode == "AE":
                    x_ = x["inp"].float().to(device)
                    reconstructed = model(x_)
                    loss = loss_function(reconstructed, x_)
                    losses_val.update(loss.item(), x_.shape[0])
                elif mode == "train":
                    x_train, y_train = batch['inp'].to(
                        device), batch['oup'].to(device)
                    output = model(x_train.float())
                    loss = loss_function(output, y_train.float())
                    losses_val.update(loss.item(), x_train.shape[0])
            print("  Validation average loss {:.3e}.".format(losses_val.avg))
            if losses_val.avg > losses_val_hist:
                val_loss_inc_epochs += 1
                print(
                    f"Detected increasing validation loss! Stop training if continuing increasing for {val_loss_inc_epochs_max - val_loss_inc_epochs} epochs!")
            else:
                val_loss_inc_epochs = 0
                model_best = deepcopy(model)
                losses_val_hist = losses_val.avg
            if val_loss_inc_epochs >= val_loss_inc_epochs_max:
                break

        if scheduler is not None:
            scheduler.step()
        if to_save:
            ckpt = {
                'epoch': epoch,
                'model_state': model_best.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
            }
            save_checkpoint(ckpt, str(mode) + 'best.pth')
    return outputs, model_best


def save_checkpoint(state, filename):
    torch.save(state, os.path.join(path, 'model' + filename))
    print("Model saved!")


def load_checkpoint(filename, model, optimizer, scheduler=None):
    ckpt = torch.load(os.path.join(path, filename))
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler'])
    print("Model at " + os.path.join(path, filename) + ' loaded.')
    return ckpt['epoch'], model, optimizer, scheduler


BATCH_SIZE = 256
ENCODING_DIM = 128
VALIDATION_PROP = 0.2
EPOCHS_AE = 250
EPOCHS_PT = 200
EPOCHS_RT = 10000


if __name__ == '__main__':
    path = os.getcwd()
    dpath = os.path.join(path, 'data')
    data = Data(dpath)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_function = nn.MSELoss()

    #############################
    # Training autoencoder
    #############################

    # Train
    # model_AE = AutoEncoder().to(device)
    # optimizer = torch.optim.Adam(model_AE.parameters(),
    #                              lr=8e-3,
    #                              weight_decay=1e-8)
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=15, gamma=0.5)

    # ae_set = MoleculeDataSet(data, Data_reduced=[], mode="AE")
    # train_len = int((1 - VALIDATION_PROP) * len(ae_set))
    # ae_t_set, ae_v_set = random_split(
    #     ae_set, [train_len, len(ae_set) - train_len])
    # ae_loader = DataLoader(ae_set, batch_size=BATCH_SIZE,
    #                        shuffle=False,
    #                        drop_last=False,
    #                        )  # for all pretrain data, prediction
    # ae_t_loader = DataLoader(
    #     ae_t_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    # ae_v_loader = DataLoader(
    #     ae_v_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # _, model_AE_best = train(
    #     model_AE, EPOCHS_AE, ae_t_loader, ae_v_loader, optimizer, scheduler, mode="AE", to_save=True)
    # model_AE = model_AE_best

    # Load model
    print("Loading AE model")
    model_AE = AutoEncoder().to(device)
    optimizer = torch.optim.Adam(model_AE.parameters(),
                                 lr=8e-3,
                                 weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=15, gamma=0.5)

    starting_epoch, model_AE, optimizer, scheduler = load_checkpoint(
        os.path.join(path, 'modelAEbest.pth'), model_AE, optimizer, scheduler)

    #############################
    # Pretraining neural-network
    #############################

    # get the reduced features
    model_AE.eval()
    ptrain_tensor = torch.from_numpy(data.pfeatures).float().to(device)
    pfeatures_reduced = model_AE.encoder(ptrain_tensor).detach().cpu().numpy()

    # train
    # pre-train 50000 data with labels
    model_nn = NeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model_nn.parameters(), lr=8e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.2)

    ptrain_reduced_set = MoleculeDataSet(data,
                                         Data_reduced=pfeatures_reduced,
                                         mode="pretrain")
    train_len = int((1 - VALIDATION_PROP) * len(ptrain_reduced_set))
    ptrain_red_t_set, ptrain_red_v_set = random_split(
        ptrain_reduced_set, [train_len, len(ptrain_reduced_set) - train_len])
    ptrain_reduced_loader = DataLoader(ptrain_reduced_set, batch_size=BATCH_SIZE,
                                       shuffle=True,
                                       drop_last=False,
                                       )
    ptrain_red_t_loader = DataLoader(
        ptrain_red_t_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    ptrain_red_v_loader = DataLoader(
        ptrain_red_v_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    _, model_nn_best = train(model_nn, EPOCHS_PT, ptrain_red_t_loader,
                             ptrain_red_v_loader, optimizer, scheduler, mode="train", to_save=True)
    model_nn = model_nn_best

    # # Load model
    # print("Loading NN model")
    # model_nn = NeuralNetwork().to(device)
    # optimizer = torch.optim.Adam(model_nn.parameters(), lr=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=10, gamma=0.2)

    # starting_epoch, model_nn, optimizer, scheduler = load_checkpoint(
    #     os.path.join(path, 'modeltrainbest.pth'), model_nn, optimizer, scheduler)

    #############################
    # Retraining neural-network
    #############################

    # get the reduced features
    model_AE.eval()
    train_tensor = torch.from_numpy(data.tfeatures).float().to(device)
    tfeatures_reduced = model_AE.encoder(train_tensor).detach().cpu().numpy()

    # retrain nn model using 100 data with target labels
    # model_nn = NeuralNetwork().to(device)
    params = [[], [], [], []]
    for name, param in model_nn.named_parameters():
        if 'layer1' in name:
            params[0].append(param)
        elif 'layer2' in name:
            params[1].append(param)
        elif 'layer3' in name:
            params[2].append(param)
        elif 'layer4' in name:
            params[3].append(param)
    optimizer = torch.optim.Adam([
        {'params': params[0], 'lr': 1e-5},
        {'params': params[1], 'lr': 1e-4},
        {'params': params[2], 'lr': 5e-4},
        {'params': params[3], 'lr': 1e-3},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.25)

    train_reduced_set = MoleculeDataSet(data,
                                        Data_reduced=tfeatures_reduced,
                                        mode="retrain")
    train_len = int((1 - VALIDATION_PROP) * len(train_reduced_set))
    train_red_t_set, train_red_v_set = random_split(
        train_reduced_set, [train_len, len(train_reduced_set) - train_len])
    # train_reduced_loader = DataLoader(train_reduced_set, batch_size=BATCH_SIZE,
    #                                   shuffle=True,
    #                                   drop_last=False,
    #                                   )
    train_red_t_loader = DataLoader(
        train_red_t_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    train_red_v_loader = DataLoader(
        train_red_v_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    outputs, model_nn_best = train(model_nn, EPOCHS_RT, train_red_t_loader,
                                   train_red_v_loader, optimizer, scheduler, mode="train")
    model_nn = model_nn_best

    #############################
    # Test
    #############################
    # get the reduced features
    model_AE.eval()
    test_tensor = torch.from_numpy(data.testfeatures).float().to(device)
    testfeatures_reduced = model_AE.encoder(test_tensor)

    # get the predictions
    prediction = model_nn(testfeatures_reduced).detach().cpu().numpy()
    IDs = data.test_ID
    with open('prediction.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'y'])
        for i in range(len(IDs)):
            writer.writerow([IDs[i], prediction[i][0]])
