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
pd.options.mode.copy_on_write = True

tqdm.pandas()

r = random.Random(7)

bound=(1,3)

# a = pd.read_csv("../investigator_nacc57.csv")
# len(a[a.NACCETPR == 88])
# len(a[(a.NACCETPR == 1) & (a.DEMENTED == 1)])
# len(a[(a.NACCETPR == 1) & (a.NACCTMCI == 1)])
# len(a[(a.NACCETPR == 1) & (a.NACCTMCI == 2)])

# loading data
class NACCCmpDataset(Dataset):

    def __init__(self, data_path, feature_path, target_indicies=[1,3,4], fold=0):
        """The NeuralPsycology Dataset

        Arguments:

        data_path (str): path to the NACC csv
        feature_path (str): path to a text file with the input features to scean
        [target_indicies] ([int]): how to translate the output key values
                                   to the indicies of an array
        [fold] (int): the n-th fold to select
        
        """

        # initialize superclass
        super(NACCCmpDataset, self).__init__()

        #### OPS ####
        # load the data
        data = pd.read_csv(data_path)

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

        #### TARGET BALANCING ####
        # crop the data to ensure balanced classes
        # TODO better dataaug that could exist?
        # we crop by future target, because that ensures
        # that results in more balanced classes for current
        # target even if we are nox explicitly balancing it
        min_class = min(data.current_target.value_counts())

        data = pd.concat([data[data.current_target==0].sample(n=min_class, random_state=7),
                          data[data.current_target==1].sample(n=min_class, random_state=7),
                          data[data.current_target==2].sample(n=min_class, random_state=7)]).sample(frac=1, random_state=7)

        self.raw = data

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

        # calculate next measurement age
        data = data.sort_values(by=["NACCAGE"])

        # subtract by the root no get age
        def subtract_by_root(item):
            return { "progression": item.NACCAGE-data[data.NACCID == item.NACCID].sort_values(by="NACCAGE").iloc[0].NACCAGE,
                     "id": item.NACCID }
        progression_metadata = data.apply(subtract_by_root, axis=1)
        progression_metadata = pd.DataFrame(progression_metadata.tolist(),
                                            index=progression_metadata.index)
        progression_metadata["id_num"] = progression_metadata.id.apply(lambda x:int(x[4:]))

        # stitch it into data
        data["DUMMY"] = np.random.randint(1, 3, data.shape[0])
        data = data[features+["DUMMY", "current_target", "NACCID", "NACCAGE"]]
        # data = data.dropna()

        def extract_pair(grp):
            if len(grp) == 1:
                return None
            res = []
            label = []
            for i in range(0, len(grp)-1):
                next_sample = r.randint(i, len(grp)-1)
                smp = grp.iloc[[i, next_sample]][features]
                if sum((smp.iloc[0] != smp.iloc[1]).to_list()) > 0:
                    if (r.choice([0,1])):
                        res.append(smp.iloc[::-1])
                        label.append(True)
                    else:
                        res.append(smp)
                        label.append(False)

            return res, label
        # sort the data by age and group, making pairwise 
        pairs = data.sort_values(["NACCAGE"]).groupby("NACCID").apply(extract_pair)

        # crop the data for validatino
        val_raw = pairs[pairs.index.isin(test_participants)].tolist()
        train_raw = pairs[pairs.index.isin(train_participants)].tolist()

        # combine to obtain raw data
        val_data, val_targets = zip(*[i for i in val_raw if i != None])
        train_data, train_targets = zip(*[i for i in train_raw if i != None])

        self.val_data = [j for i in val_data for j in i]
        self.val_targets = [j for i in val_targets for j in i]

        self.train_data = [j for i in train_data for j in i]
        self.train_targets = [j for i in train_targets for j in i]

        self.val_raw = data[data.NACCID.isin(test_participants)][features+["NACCID", "NACCAGE"]]
        self.train_raw = data[data.NACCID.isin(train_participants)][features+["NACCID", "NACCAGE"]]

        # self.val_targets = data[data.NACCID.isin(test_participants)].current_target
        # self.val_data_raw = data[data.NACCID.isin(test_participants)]
        # self.val_data_prog = progression_metadata[progression_metadata.id.isin(test_participants)]

        # self.data = data[data.NACCID.isin(train_participants)][features]
        # self.targets = data[data.NACCID.isin(train_participants)].current_target
        # self.data_raw = data[data.NACCID.isin(train_participants)]
        # self.data_prog = progression_metadata[progression_metadata.id.isin(train_participants)]

        # self.features = features

    def __process(self, data):
        # as a test, we report results without masking
        # if a data entry is <0 or >80, it is "not found"
        # so, we encode those values as 0 in the FEATURE
        # column, and encode another feature of "not-found"ness
        data_found = (data > 80) | (data < 0)
        data[data_found] = 0
        # then, the found-ness becomes a mask
        data_found_mask = data_found
        # don't attend to dummy 
        data_found_mask[-1] = True

        # if it is a sample with no tangible data
        # well give up and get another sample:
        if sum(~data_found_mask) == 0:
            # if not index:
            raise ValueError("All-Zero Found!")
            # indx = random.randint(2,5)
            # if index-indx <= 0:
            #     return self[index+indx]
            # else:
            #     return self[index-indx]
        
        # # seed the one-hot vector
        # one_hot_target = [0 for _ in range(3)]
        # # and set it
        # one_hot_target[target] = 1

        return torch.tensor(data).float()/30, torch.tensor(data_found_mask).bool()

    def __getitem__(self, index):
        # index the data
        data = self.train_data[index].copy()
        tgt = self.train_targets[index]

        target = [0 for _ in range(2)]
        if tgt:
            target[1] = 1
        else:
            target[0] = 1

        d1, m1 = self.__process(data.iloc[0])
        d2, m2 = self.__process(data.iloc[1])

        return torch.stack([d1, d2]), torch.stack([m1, m2]), torch.tensor(target)

    @functools.cache
    def val(self):
        """Return the validation set"""

        # collect dataset
        dataset = []

        print("Processing validation data...")

        # get it
        for index in tqdm(range(len(self.val_data))):
            data = self.val_data[index].copy()
            tgt = self.val_targets[index]

            target = [0 for _ in range(2)]
            if tgt:
                target[1] = 1
            else:
                target[0] = 1

            d1, m1 = self.__process(data.iloc[0])
            d2, m2 = self.__process(data.iloc[1])

            try:
                dataset.append((torch.stack([d1, d2]), torch.stack([m1, m2]), torch.tensor(target)))
            except ValueError:
                continue # all zero ignore

        # return parts
        inp, mask, target, = zip(*dataset)

        # process already divides by 30; don't do it twice
        return (torch.stack(inp).float(), torch.stack(mask).bool(),
                torch.stack(target).float())

    def __len__(self):
        return len(self.train_data)

# d = NACCCmpDataset("./investigator_nacc57.csv", "./features/combined")
# tmp = d.val()
# tmp[0].shape
# tmp[2].shape
# # sum(d.train_targets)/len(d.train_targets)
