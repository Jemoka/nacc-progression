# modeling libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

import gc

import wandb

import functools

from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import KFold
from sklearn.utils import shuffle

# nicies
from tqdm import tqdm

# stdlib utilities
import json
import glob
import math
import random
import pickle

tqdm.pandas()

import random
bound=(1,3)

R = random.Random(7)

# a = pd.read_csv("../investigator_nacc57.csv")
# len(a[a.NACCETPR == 88])
# len(a[(a.NACCETPR == 1) & (a.DEMENTED == 1)])
# len(a[(a.NACCETPR == 1) & (a.NACCTMCI == 1)])
# len(a[(a.NACCETPR == 1) & (a.NACCTMCI == 2)])

class NACCLongitudinalDataset(Dataset):

    def __init__(self, data_path, feature_path,
              # skipping 2 impaired because of labeling inconsistency
                 target_indicies=[1,3,4], fold=0, one_to_three=False):
        """The NeuralPsycology Dataset

        Arguments:

        data_path (str): path to the NACC csv
        feature_path (str): path to a text file with the input features to scean
        [target_indicies] ([int]): how to translate the output key values
                                   to the indicies of an array
        [fold] (int): the n-th fold to select
        """

        # initialize superclass
        super(NACCLongitudinalDataset, self).__init__()

        #### OPS ####
        # load the data
        data = pd.read_csv(data_path)
        self.raw = data

        # get the fature variables
        with open(feature_path, 'r') as f:
            lines = f.readlines()
            features = list(sorted(set([i.strip() for i in lines])))

        #### CURRENT PREDICTION TARGETS ####
        # construct the artificial target 
        # everything is to be ignored by default
        # this new target has: 0 - Control; 1 - MCI; 2 - Dementia
        data.loc[:, "current_target"] = -1

        # NACCETPR == 88; DEMENTED == 0 means Control
        data.loc[(data.NACCETPR == 88)&
                (data.DEMENTED == 0), "current_target"] = 0
        # NACCETPR == 1; DEMENTED == 1; means AD Dementia
        data.loc[(data.NACCETPR == 1)&
                (data.DEMENTED == 1), "current_target"] = 2
        # NACCETPR == 1; DEMENTED == 0; NACCTMCI = 1 or 2 means amnestic MCI
        data.loc[((data.NACCETPR == 1)&
                (data.DEMENTED == 0)&
                ((data.NACCTMCI == 1) |
                (data.NACCTMCI == 2))), "current_target"] = 1

        # drop the columns that are irrelavent to us (i.e. not the labels above)
        data = data[data.current_target != -1]
        data = data[data.NACCAGE > 65]
        # data.current_target.value_counts()

        #### FUTURE PREDICTION TARGETS ####
        def process_participant(part, converted=False):
            if len(part) <= 1:
                return None
            sorted = part.sort_values(by=["NACCAGE"])
            crops = list(range(len(sorted)))[1:]
            # crops = R.sample(possible_crops, R.randint(1, len(possible_crops)))

            if converted:
                res = [(sorted.iloc[:j], sorted.iloc[j].current_target,
                        # sorted.iloc[:j+1].NACCAGE-sorted.iloc[0].NACCAGE) for j in crops
                        sorted.iloc[:j+1].NACCAGE-65) for j in crops
                    if sorted.iloc[j-1].current_target > sorted.iloc[j].current_target]
            else:
                res = [(sorted.iloc[:j], sorted.iloc[j].current_target,
                        # sorted.iloc[:j+1].NACCAGE-sorted.iloc[0].NACCAGE) for j in crops
                        sorted.iloc[:j+1].NACCAGE-65) for j in crops
                    if sorted.iloc[j-1].current_target <= sorted.iloc[j].current_target]


            if one_to_three:
                res = [i for i in res
                       if (i[2].iloc[-1] - i[2].iloc[0]) <= 3]

            if len(res) == 0:
                return None

            return res

        data = data[features+["current_target", "NACCID", "NACCAGE"]]
        data = data.dropna()

        res_data_converted = data.groupby(data.NACCID).apply(lambda x:process_participant(x, True))
        res_data_not_converted = data.groupby(data.NACCID).apply(lambda x:process_participant(x, False))
        # filter out for blanks
        res_data_converted = [j for i in res_data_converted if i for j in i]
        res_data_not_converted = [j for i in res_data_not_converted if i for j in i]

        # balance for counts of each
        # data_count = min(len(res_data_converted), len(res_data_not_converted))
        # res_data = (R.sample(res_data_converted, data_count) +
                    # R.sample(res_data_not_converted, data_count))
        res_data = res_data_converted+res_data_not_converted

        # compute sample of each type
        control_samples = []
        mci_samples = []
        dementia_samples = []

        for elem in res_data:
            i,j,k = elem
            if j == 0:
                control_samples.append((i,j, k))
            elif j == 1:
                mci_samples.append((i,j, k))
            elif j == 2:
                dementia_samples.append((i,j,k))

        # min elements to sample from each class
        num_samples = min(len(mci_samples),
                          len(dementia_samples),
                          len(control_samples))
        res_data = (R.sample(mci_samples, num_samples) +
                            R.sample(dementia_samples, num_samples) +
                            R.sample(control_samples, num_samples))
        R.shuffle(res_data)

        #### TRAIN_VAL SPLIT ####
        kf = KFold(n_splits=10, shuffle=True, random_state=7)

        # split participants for indexing
        participants = list(sorted(set(data.NACCID.tolist())))
        splits = kf.split(participants)
        train_ids, test_ids = list(splits)[fold]
        train_participants = [participants[i] for i in train_ids]
        test_participants = [participants[i] for i in test_ids]

        # calculate number of features
        self._num_features = len(features)
        # we add 1 for dummy variable used for fine tuning later

        # crop the data for validatino
        self.val_data = [i[0][features] for i in res_data if i[0].NACCID.iloc[0] in test_participants]
        self.val_targets = [i[1] for i in res_data if i[0].NACCID.iloc[0] in test_participants]
        self.val_temporal = [i[2] for i in res_data if i[0].NACCID.iloc[0] in test_participants]

        self.data = [i[0][features] for i in res_data if i[0].NACCID.iloc[0] in train_participants]
        self.targets = [i[1] for i in res_data if i[0].NACCID.iloc[0] in train_participants]
        self.temporal = [i[2] for i in res_data if i[0].NACCID.iloc[0] in train_participants]

    def __process_sample(self, data):
        data = data.copy()
        # the discussed dataprep
        # if a data entry is <0 or >80, it is "not found"
        # so, we encode those values as 0 in the FEATURE
        # column, and encode another feature of "not-found"ness
        data_found = (data > 80) | (data < 0)
        data.iloc[data_found] = 0
        # then, the found-ness becomes a mask
        data_found_mask = data_found

        return torch.tensor(data).float()/30, torch.tensor(data_found_mask).bool()

    def __process(self, data, target, temporal, index=None):
        # iterate through each column of the data to 
        datas, masks = zip(*[self.__process_sample(i) for _, i in data.iterrows()])
        datas = torch.stack(datas)
        masks = torch.stack(masks)

        # get the time invariant samples as a seperate set (data 1)
        data_inv = datas[-1].clone()
        data_inv_mask = masks[-1].clone()

        # and mask out the time invariant data from timeseries
        data_var = (datas.clone()-data_inv)[:-1]
        data_var_mask = masks[:-1]

        # seed the one-hot vector
        one_hot_target = [0 for _ in range(3)]
        # and set it
        one_hot_target[int(target)] = 1
        times = torch.tensor(temporal.tolist()).float()
        # second to last sample is our inv sample; last is our target
        temporal = times[:-2]

        return data_inv, data_inv_mask, data_var, data_var_mask, temporal, times[-1], one_hot_target

    def __getitem__(self, index):
        # index the data
        data = self.data[index]
        target = self.targets[index]
        temporal = self.temporal[index]

        di, dim, dv, dvm, tp,tm, out = self.__process(data, target, temporal, index)
        return di, dim, dv, dvm, tp,tm, out

    @functools.cache
    def val(self):
        """Return the validation set"""

        # collect dataset
        dataset = []

        print("Processing validation data...")

        # get it
        for index in tqdm(range(len(self.val_data))):
            try:
                dataset.append(self.__process(self.val_data[index],
                                              self.val_targets[index],
                                              self.val_temporal[index]))
            except ValueError:
                continue # all zero ignore

        # return parts
        di, dim, dv, dvm, tp, tm, out = zip(*dataset)

        # process already divides by 30; don't do it twice
        return di, dim, dv, dvm, tp, tm, out

    def __len__(self):
        return len(self.data)

# from scipy.stats import pearsonr

# dataset = NACCLongitudinalDataset("./investigator_nacc57.csv", "./features/combined")

# results = []
# for i in tqdm(dataset):
#     di, _, dv, *res = i

#     if dv.size(0) < 4:
#         continue

#     split = dv.split(2)
#     split = split if split[-1].shape[0] == split[0].shape[0] else split[:-1]
#     split = torch.stack(split)

#     normalized_difference = (split[:,1,:]-split[:,0,:]) / (dv.std(dim=-2) + 1e-12)
#     split = normalized_difference.split(2)
#     split = split if split[-1].shape[0] == split[0].shape[0] else split[:-1]
#     split = torch.stack(split)

#     final_difference = (split[:,1,:]-split[:,0,:])
#     mean_difference = final_difference.mean().item()
#     prediction = res[-1].index(1)

#     results.append([mean_difference, prediction])

# len(results) # 5795 / 22640 had more than 4 samples

# result = pearsonr([i[0] for i in results],
#                   [i[1] for i in results])
# result.pvalue # yes 0.0442

# result = pearsonr([i[0] for i in results],
#                   [i[1] == 2 for i in results])
# result.pvalue # yes 0.079

# result = pearsonr([i[0] for i in results],
#                   [i[1] == 0 for i in results])
# result.pvalue # yes 0.075

# result = pearsonr([i[0] for i in results],
#                   [i[1] == 1 for i in results])
# result.pvalue # no 0.843

# sum(i[1] == 1 for i in results) # yet this is 1702


