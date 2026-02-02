import numpy as np
import torch
from torch.utils.data import Dataset


class Load_Dataset(Dataset):
    def __init__(self, dataset):
        super(Load_Dataset, self).__init__()
        X_train = dataset["samples"]
        y_train = dataset["labels"]
        print(f'This dataset has {max(y_train) + 1} classes')

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train).to('cuda')
            self.y_data = torch.from_numpy(y_train).long().to('cuda')
        else:
            self.x_data = X_train.float().to('cuda')
            self.y_data = y_train.long().to('cuda')
        self.len = X_train.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
