from __future__ import print_function, division
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from sklearn.decomposition import PCA


class H36MDataset(Dataset):

    def __init__(self, fname, normalize_2d=True,
                 get_2dgt=False, subjects=[1, 5, 6, 7, 8], get_PCA=False, normalize_func=None):

        joints = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 25, 26, 27, 17, 18, 19]

        pickle_off = open(fname, "rb")
        loaddata = pickle.load(pickle_off)

        # select subjects
        selection_array = np.zeros(len(loaddata['subject']), dtype=bool)
        for s in subjects:
            selection_array = np.logical_or(selection_array, (np.array(loaddata['subject']) == s))

        self.data = dict()
        self.data['poses_3d'] = loaddata['poses_3d'][selection_array][:, joints, :].transpose(0, 2, 1).reshape(-1, 3*len(joints))
        self.data['poses_3d'] = torch.tensor(self.data['poses_3d'], dtype=torch.float)

        if get_2dgt == True:
            self.data['poses_2d'] = loaddata['poses_2d'][selection_array][:, joints, :].transpose(0, 2, 1).reshape(-1, 2*len(joints))
        else:
            self.data['poses_2d'] = loaddata['poses_2d_pred'][selection_array]

        self.data['poses_2d'] = torch.tensor(self.data['poses_2d'], dtype=torch.float)

        self.normalize_2d = normalize_2d
        self.get_2dgt = get_2dgt

        if normalize_func:
            self.data['poses_2d'] = normalize_func(self.data['poses_2d'])
            #self.data['poses_2d_pred'] = normalize_func(self.data['poses_2d_pred'])
        else:
            if self.normalize_2d:
                self.data['poses_2d'] = (
                        self.data['poses_2d'].reshape(-1, 2, 16) -
                        self.data['poses_2d'].reshape(-1, 2, 16).mean(axis=2, keepdims=True)).reshape(-1, 32)
                self.data['poses_2d'] /= np.linalg.norm(self.data['poses_2d'], ord=2, axis=1, keepdims=True)

        if get_PCA:
            self.pca = PCA()
            self.pca.fit(self.data['poses_2d'])

    def __len__(self):
        return self.data['poses_3d'].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()

        if self.get_2dgt:
            sample['p2d_gt'] = self.data['poses_2d'][idx]
        else:
            sample['p2d_pred'] = self.data['poses_2d'][idx]

        sample['poses_3d'] = self.data['poses_3d'][idx]

        return sample

