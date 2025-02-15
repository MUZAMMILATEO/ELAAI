import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt

import numpy as np


class RaFDDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.paths[index]
        # x, y, x_gray, y_gray = pkload(path)
        # MK UPDATE ------------------------
        data_dict = pkload(path)
        x, y, x_gray, y_gray = data_dict['fixed_image'], data_dict['moving_image'], data_dict['fixed_mask'], data_dict['moving_mask']
        # ## ###### ------------------------
        x, y           = x[None, ...], y[None, ...]
        x_gray, y_gray = x_gray[None, ...], y_gray[None, ...]
        # x_gray, y_gray = self.transforms([x_gray, y_gray])
        x = np.ascontiguousarray(x.astype(np.float32))
        y = np.ascontiguousarray(y.astype(np.float32))
        x_gray = np.ascontiguousarray(x_gray.astype(np.float32))
        y_gray = np.ascontiguousarray(y_gray.astype(np.float32))
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        x_gray, y_gray = torch.from_numpy(x_gray), torch.from_numpy(y_gray)
        # plt.figure()
        # xplt = x.squeeze(0)
        # xplt = xplt.unsqueeze(2)
        # xplt = xplt/xplt.max()
        # plt.imshow(torch.cat((xplt, xplt, xplt), axis=2), cmap='gray')
        # plt.show()
        return x, y, x_gray, y_gray

    def __len__(self):
        return len(self.paths)


class RaFDInferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        # x, y, x_gray, y_gray = pkload(path)
        # MK UPDATE ------------------------
        data_dict = pkload(path)
        x, y, x_gray, y_gray = data_dict['fixed_image'], data_dict['moving_image'], data_dict['fixed_mask'], data_dict['moving_mask']
        # ## ###### ------------------------
        x, y = x[None, ...], y[None, ...]
        x_gray, y_gray = x_gray[None, ...], y_gray[None, ...]
        x_gray = np.ascontiguousarray(x_gray.astype(np.float32))
        y_gray = np.ascontiguousarray(y_gray.astype(np.float32))
        x = np.ascontiguousarray(x.astype(np.float32))
        y = np.ascontiguousarray(y.astype(np.float32))
        x_gray, y_gray = torch.from_numpy(x_gray), torch.from_numpy(y_gray)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y, x_gray, y_gray

    def __len__(self):
        return len(self.paths)