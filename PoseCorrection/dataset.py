import pickle
import itertools
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch_dct as dct

from PoseCorrection.softdtw import SoftDTW


class HV3D(Dataset):

    def __init__(self, data_path, dct_n=25, split=0, sets=None, is_cuda=False, add_data=None):
        if sets is None:
            sets = [[0, 1], [2], [3]]

        self.dct_n = dct_n

        correct, other = load_data(data_path, sets[split], add_data=add_data)
        pairs = dtw_pairs(correct, other, is_cuda=is_cuda)

        self.targets_label = [i[1] for i in pairs]
        self.inputs_label = [i[0] for i in pairs]
        self.targets = [correct[i] for i in self.targets_label]
        self.inputs_raw = [other[i] for i in self.inputs_label]
        self.inputs = [dct.dct_2d(torch.from_numpy(x))[:, :self.dct_n].numpy() if x.shape[1] >= self.dct_n else
                       dct.dct_2d(torch.nn.ZeroPad2d((0, self.dct_n - x.shape[1], 0, 0))(torch.from_numpy(x))).numpy()
                       for x in self.inputs_raw]

        self.node_n = np.shape(self.inputs_raw[0])[0]
        self.batch_ids = list(range(len(self.inputs_raw)))

    def __len__(self):
        return np.shape(self.inputs)[0]

    def __getitem__(self, item):
        return self.batch_ids[item], self.inputs[item]


def load_data(data_path, subs, add_data=None):
    with open(data_path, "rb") as f:
        data_gt = pickle.load(f)

    if add_data is not None:
        with open(add_data, "rb") as f:
            data = pickle.load(f)
        labels = pd.DataFrame(data['labels'], columns=['act', 'sub', 'lab', 'rep', 'cam'])
    else:
        data = data_gt
        labels = pd.DataFrame(data['labels'], columns=['act', 'sub', 'lab', 'rep', 'frame'])
        labels['cam'] = 'gt'

    joints = list(range(15)) + [19, 21, 22, 24]

    labels_gt = pd.DataFrame(data_gt['labels'], columns=['act', 'sub', 'lab', 'rep', 'frame'])
    labels_gt['cam'] = 'gt'

    labels[['lab', 'rep']] = labels[['lab', 'rep']].astype(int)
    labels_gt[['lab', 'rep']] = labels_gt[['lab', 'rep']].astype(int)

    subs = labels[['act', 'sub', 'lab', 'rep']].drop_duplicates().groupby('sub').count().rep[subs]
    indices = labels['sub'].isin(subs.index)
    indices_gt = labels_gt['sub'].isin(subs.index)
    labels = labels[indices]
    labels_gt = labels_gt[indices_gt]

    lab1 = labels_gt[labels_gt['lab'] == 1].groupby(['act', 'sub', 'lab', 'rep', 'cam']).groups
    labnot1 = labels.groupby(['act', 'sub', 'lab', 'rep', 'cam']).groups

    poses = data['poses'][:, :, joints]
    poses_gt = data_gt['poses'][:, :, joints]

    correct = {k: poses_gt[v].reshape(-1, poses_gt.shape[1] * poses_gt.shape[2]).T for k, v in lab1.items()}
    other = {k: poses[v].reshape(-1, poses.shape[1] * poses.shape[2]).T for k, v in labnot1.items()}

    return correct, other


def dtw_pairs(correct, incorrect, is_cuda=False):
    pairs = []
    for act, sub in set([(k[0], k[1]) for k in incorrect.keys()]):
        correct_sub = {k: v for k, v in correct.items() if k[0] == act and k[1] == sub}
        incorrect_sub = {k: v for k, v in incorrect.items() if k[0] == act and k[1] == sub}
        dtw_sub = {k: {} for k in incorrect_sub.keys()}
        for i, pair in enumerate(itertools.product(incorrect_sub, correct_sub)):
            criterion = SoftDTW(use_cuda=is_cuda, gamma=0.01)
            if is_cuda:
                p0 = torch.from_numpy(np.expand_dims(incorrect_sub[pair[0]].T, axis=0)).cuda()
                p1 = torch.from_numpy(np.expand_dims(correct_sub[pair[1]].T, axis=0)).cuda()
            else:
                p0 = torch.from_numpy(np.expand_dims(incorrect_sub[pair[0]].T, axis=0))
                p1 = torch.from_numpy(np.expand_dims(correct_sub[pair[1]].T, axis=0))
            dtw_sub[pair[0]][pair[1]] = (criterion(p0, p1) - 1 / 2 * (criterion(p0, p0) + criterion(p1, p1))).item()
        dtw = pd.DataFrame.from_dict(dtw_sub, orient='index').idxmin(axis=1)
        pairs = pairs + list(zip(dtw.index, dtw))
    return pairs


def dtw_pairs_4targ(correct, incorrect, is_cuda=False, test=False):
    pairs = []
    for sub in set([k[1] for k in correct.keys()]):
        dtw_sub = {k: {} for k in incorrect.keys()}
        if test:
            correct_sub = correct
        else:
            correct_sub = {k: v for k, v in correct.items() if k[1] == sub}
        for i, pair in enumerate(itertools.product(incorrect, correct_sub)):
            criterion = SoftDTW(use_cuda=is_cuda, gamma=0.01)
            if is_cuda:
                p0 = torch.from_numpy(np.expand_dims(incorrect[pair[0]].T, axis=0)).cuda()
                p1 = torch.from_numpy(np.expand_dims(correct_sub[pair[1]].T, axis=0)).cuda()
            else:
                p0 = torch.from_numpy(np.expand_dims(incorrect[pair[0]].T, axis=0))
                p1 = torch.from_numpy(np.expand_dims(correct_sub[pair[1]].T, axis=0))
            dtw_sub[pair[0]][pair[1]] = (criterion(p0, p1) - 1 / 2 * (criterion(p0, p0) + criterion(p1, p1))).item()
        dtw = pd.DataFrame.from_dict(dtw_sub, orient='index').idxmin(axis=1)
        pairs = pairs + list(zip(dtw.index, dtw))
        if test:
            return pairs
    return pairs
