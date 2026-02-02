import argparse
import time
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from torch import optim
import os

from Dataloader import Load_Dataset
from models import *
from torch.utils.data import Dataset
import math

from utils import Sim, Dis

parser = argparse.ArgumentParser()

parser.add_argument('--epoches', default='100', type=int)
parser.add_argument('--data_path', default='data/HAR/', type=str)
parser.add_argument('--alpha1', default=.1, type=float)
parser.add_argument('--alpha2', default=.1, type=float)
parser.add_argument('--batchsize', default=258, type=int)
parser.add_argument('--lr', default=5e-3, type=float)
parser.add_argument('--down_sampling_layers', default=3, type=int)
parser.add_argument('--channel', default=128, type=int)
parser.add_argument('--temporal_size', default=128, type=int)


class EarlyStopping:
    def __init__(self, patience=15, verbose=False, dump=False, ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_ls = None
        self.best_model = Model(args).to('cuda')
        self.early_stop = False
        self.dump = dump
        self.val_loss_min = np.Inf
        self.delta = 0
        self.trace_func = print

    def __call__(self, val_loss, model, epoch):
        ls = val_loss
        if self.best_ls is None:
            self.best_ls = ls
            self.best_model.load_state_dict(model.state_dict())
        elif ls > self.best_ls + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} with {epoch}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_ls = ls
            self.best_model.load_state_dict(model.state_dict())
            self.counter = 0


def data_generator(args):
    train_dataset = torch.load(os.path.join(args.data_path, "train.pt"))
    val_dataset = torch.load(os.path.join(args.data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(args.data_path, "test.pt"))

    train_dataset = Load_Dataset(train_dataset)
    val_dataset = Load_Dataset(val_dataset)
    test_dataset = Load_Dataset(test_dataset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batchsize,
                                               shuffle=True, drop_last=True,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batchsize,
                                             shuffle=False, drop_last=False,
                                             num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batchsize,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, val_loader, test_loader


def Test(test_loader, bestmodel):
    predicts = None
    labels = None
    bestmodel.eval()
    for data, label in test_loader:
        out, tduplicate, tunique = bestmodel(data)
        out = torch.argmax(out.cpu().detach(), dim=-1).reshape(-1)
        label = label.cpu().detach().reshape(-1)

        if predicts == None:
            predicts = out
            labels = label
        else:
            predicts = torch.cat([predicts, out])
            labels = torch.cat([labels, label])
    acc = accuracy_score(predicts, labels) * 100
    f1 = f1_score(predicts, labels, average="macro") * 100
    mcc = matthews_corrcoef(predicts, labels)
    printtext = "ACC:{:.2f}".format(acc) + '% ' + 'F1:{:.2f}'.format(
        f1) + '% ' + 'MCC:{:.4f}'.format(
        mcc)
    print(printtext)
    return acc, f1, mcc


def Trainer(args, model, optimizer, train_loader, val_loader, test_loader, early_stopping, scheduler=None):
    PredLossFun = nn.CrossEntropyLoss()
    for epoch in range(1, args.epoches + 1):
        train_losses, val_losses, test_losses = [], [], []
        model.train()
        for data, label in train_loader:
            simloss = Sim()
            disloss = Dis()
            optimizer.zero_grad()
            out, tduplicate, tunique = model(data)
            loss = PredLossFun(out, label) + args.alpha1 * simloss(tduplicate) + args.alpha2 * disloss(tunique)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            for data, label in val_loader:
                simloss = Sim()
                disloss = Dis()
                out, tduplicate, tunique = model(data)
                loss = PredLossFun(out, label) + args.alpha1 * simloss(tduplicate) + args.alpha2 * disloss(tunique)
                val_losses.append(loss.item())

        with torch.no_grad():
            model.eval()
            for data, label in test_loader:
                simloss = Sim()
                disloss = Dis()
                out, tduplicate, tunique = model(data)
                loss = PredLossFun(out, label) + args.alpha1 * simloss(tduplicate) + args.alpha2 * disloss(tunique)

                test_losses.append(loss.item())
        print('epoch:{0:}, train_loss:{1:.5f}, val_loss:{2:.5f}, test_loss:{3:.5f}'.format(epoch,
                                                                                           np.mean(train_losses),
                                                                                           np.mean(val_losses),
                                                                                           np.mean(test_losses)))

        scheduler.step(np.mean(val_losses))
        early_stopping(np.mean(val_losses), model, epoch)
        if early_stopping.early_stop:
            print("Early stopping with best_ls:{}".format(early_stopping.best_ls))
            break
        if np.isnan(np.mean(val_losses)) or np.isnan(np.mean(train_losses)):
            break

        Test(test_loader, early_stopping.best_model)


def main(args):
    # SEED = 0
    # torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(SEED)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, test_loader = data_generator(args)
    model = Model(args).float().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(.9, .99), weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=.1, patience=5)
    early_stopping = EarlyStopping(patience=15, verbose=True, dump=False)
    Trainer(args, model, optimizer, train_loader, val_loader, test_loader, early_stopping, scheduler)

    acc, f1, mcc = Test(test_loader, early_stopping.best_model)
    return acc, f1, mcc


if __name__ == '__main__':
    args = parser.parse_args()
    start = time.time()
    acc, f1, mcc = main(args)
    print(acc, f1, mcc)
