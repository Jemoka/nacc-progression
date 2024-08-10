# modeling libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import pandas as pd

import gc

import wandb

import functools

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold

# import tqdm auto
from tqdm.auto import tqdm
tqdm.pandas()

# nicies
from tqdm import tqdm

# stdlib utilities
import json
import glob
import math
import random

# model
from model import NACCModel
from model_lstm import NACCLSTMModel
from model_latefuse import NACCFuseModel
# from model_legacy import NACCModel
from dataset import *


CONFIG = {
    "fold": 0,
    # "featureset": "neuralpsych-v2",
    "featureset": "combined",
    "batch_size": 8,
    "lr": 0.000005,
    "epochs": 55,

    "nlayers": 3,
    "hidden": 2048,
    "type": "fuse",
    "one_to_three": True,
}


ONLINE = False
# ONLINE = True

run = wandb.init(project="nacc-temporal",
                 entity="jemoka", config=CONFIG, mode=("online" if ONLINE else "disabled"))

config = run.config

BATCH_SIZE = config.batch_size
LR = config.lr
EPOCHS = config.epochs

FOLD = config.fold
FEATURESET = config.featureset
# MODEL = config.base

# initialize the device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')

dataset = NACCLongitudinalDataset("./investigator_nacc57.csv",
                                  f"./features/{FEATURESET}", fold=FOLD,
                                  one_to_three=config.one_to_three)

def collate_fn(data):
    di, dim, dv, dvm, tp, tm, out = zip(*data)

    # invariant data can just be stacked
    inv_data = torch.stack(di)
    inv_mask = torch.stack(dim)
    out = torch.stack([torch.tensor(i).float() for i in out])

    # get the batch's maximum length
    time_max = max(len(i) for i in tp)
    to_pad = [time_max-i.shape[0] for i in dv]
    # pad the data and mask tensors
    var_data = torch.stack([F.pad(i, (0,0,0,j), "constant", 0) for i,j in zip(dv, to_pad)])
    var_mask = torch.stack([F.pad(i, (0,0,0,j), "constant", True) for i,j in zip(dvm, to_pad)])
    timestamps = torch.stack([F.pad(i, (0,j), "constant", 0) for i,j in zip(tp, to_pad)])
    # calculate which of the samples is padding only
    is_pad = var_mask.all(dim=2)

    return inv_data, inv_mask, var_data, var_mask, timestamps, is_pad, torch.stack(tm), out

class ValDataset(Dataset):
    def __init__(self, dataset):
        self.raw = list(zip(*dataset.val()))

    def __getitem__(self, indx):
        return self.raw[indx]

    def __len__(self):
        return len(self.raw)

validation_set = ValDataset(dataset)
validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# a = next(iter(dataloader))
# a

# if not MODEL:
if config.type.lower() == "transformer":
    model = NACCModel(3, nlayers=config.nlayers, hidden=config.hidden).to(DEVICE)
elif config.type.lower() == "lstm":
    model = NACCLSTMModel(3, nlayers=config.nlayers, hidden=config.hidden).to(DEVICE)
elif config.type.lower() == "fuse":
    model = NACCFuseModel(num_classes=3, num_features=dataset[0][0].shape[0],
                          nlayers=config.nlayers, hidden=config.hidden).to(DEVICE)

 
# else:
#     model = NACCModel(dataset._num_features, 3, nlayers=config.nlayers, hidden=config.hidden).to(DEVICE)
#     model.load_state_dict(torch.load(os.path.join(f"./models/{MODEL}", "model.save"), map_location=DEVICE))

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
# scheduler = StepLR(optimizer, step_size=8, gamma=0.75)

run.watch(model)


# get a random validation batch
def val_batch():
    return next(iter(validation_loader))

def val():
    model.eval()
    batch = [i.to(DEVICE) for i in val_batch()]

    try:
        output = model(*batch)
        run.log({"val_loss": output["loss"].detach().cpu().item()})
    except RuntimeError:
        pass
    finally:
        model.train()

# calculate the f1 from tensors
def tensor_metrics(logits, labels):
    label_indicies = np.argmax(labels, 1)
    logits_indicies  = logits

    class_names = ["Control", "MCI", "Dementia"]

    pr_curve = wandb.plot.pr_curve(label_indicies, logits_indicies, labels = class_names)
    roc = wandb.plot.roc_curve(label_indicies, logits_indicies, labels = class_names)
    cm = wandb.plot.confusion_matrix(
        y_true=np.array(label_indicies), # can't labels index by scalar tensor
        probs=logits_indicies,
        class_names=class_names
    )

    acc = sum(label_indicies == np.argmax(logits_indicies, axis=1))/len(label_indicies)

    return pr_curve, roc, cm, acc

def future_metrics(logits, labels, current_targets):
    label_indicies = np.argmax(labels, 1)
    current_target_indicies = np.argmax(current_targets, 1)
    logits_indicies = logits

    indicies_didnt_change = label_indicies == current_target_indicies

    logits_changed = logits[~indicies_didnt_change]
    labels_changed = labels[~indicies_didnt_change]

    logits_unchanged = logits[indicies_didnt_change]
    labels_unchanged = labels[indicies_didnt_change]

    future_group_changed = tensor_metrics(logits_changed, labels_changed)
    future_group_unchanged = tensor_metrics(logits_unchanged, labels_unchanged)

    return future_group_changed, future_group_unchanged

model.train()
for epoch in range(EPOCHS):
    print(f"Currently training epoch {epoch}...")

    for i, batch in tqdm(enumerate(iter(dataloader)), total=len(dataloader)):
        if i % 64 == 0:
            val()

        batchp = batch
        # send batch to GPU if needed
        batch = [i.to(DEVICE) for i in batch]

        # run with actual backprop
        output = model(*batch)

        # backprop
        try:
            output["loss"].backward()
        except RuntimeError:
            optimizer.zero_grad()
            continue
        optimizer.step()
        optimizer.zero_grad()

        # logging
        run.log({"loss": output["loss"].detach().cpu().item()})

    # scheduler.step()

# model.eval()

# we track logits and labels and count them
# finally together eventually
logits = np.empty((0,3))
labels = np.empty((0,3))
current_targets = np.empty((0,3))

print("Validating...")

try:
    # validation is large, so we do batches
    for i in tqdm(iter(validation_loader)):
        batch = [j.to(DEVICE) for j in i]
        output = model(*batch)

        # append to talley
        logits = np.append(logits, output["logits"].detach().cpu().numpy(), 0)
        labels = np.append(labels, i[-1].numpy(), 0)
except:
    breakpoint()


try:
    prec_recc, roc, cm, acc = tensor_metrics(logits, labels)
    run.log({"val_prec_recc": prec_recc,
             "val_confusion": cm,
             "val_roc": roc,
             "val_acc": acc})
except ValueError:
    breakpoint()

# Saving
print("Saving model...")
os.mkdir(f"./models/{run.name}")
torch.save(model.state_dict(), f"./models/{run.name}/model.save")
torch.save(optimizer, f"./models/{run.name}/optimizer.save")

# breakpoint()
