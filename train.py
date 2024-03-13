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
# from model_legacy import NACCModel
from dataset import *


CONFIG = {
    "fold": 0,
    "base": "efficient-dragon-18",
    "featureset": "combined",
    # "task": "future",
    "task": "current",
    "batch_size": 32,
    "lr": 0.0001,
    # "batch_size": 8,
    # "lr": 0.00005,
    "epochs": 55,

    "nlayers": 3,
    "hidden": 128,
}


TASK = CONFIG["task"]

ONE_SHOT = True
# ONE_SHOT = False
ONLINE = False
# ONLINE = True

run = wandb.init(project="nacc-progression", entity="jemoka", config=CONFIG, mode=("online" if ONLINE else "disabled"))

config = run.config

BATCH_SIZE = config.batch_size
LR = config.lr
EPOCHS = config.epochs

FOLD = config.fold
FEATURESET = config.featureset

BASE = config.base

# initialize the device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')

dataset = NACCCmpDataset("./investigator_nacc57.csv", f"./features/{FEATURESET}")

validation_set = TensorDataset(*dataset.val())
validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = NACCModel(dataset._num_features, nlayers=config.nlayers, hidden=config.hidden).to(DEVICE)

# load preload model
preload = torch.load(f"./pretrain/{BASE}/model.save", map_location=torch.device('cpu'))
# inject encoder
inject_state_dict = {i:j for i,j in preload.items() if "encoder" in i}
model.load_state_dict(inject_state_dict, strict=False)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

# get a random validation batch
def val_batch():
    start = random.randint(0, len(validation_set)//BATCH_SIZE)*BATCH_SIZE
    end = start+BATCH_SIZE
    batch = validation_set[start:end]
    return batch

def val():
    model.eval()
    with torch.inference_mode():
        batch = [i.to(DEVICE) for i in val_batch()]

        try:
            output = model(batch[0].float(), batch[1], batch[2], detach_transformer=True)
            accuracy = sum(output["logits"].argmax(1) == batch[2].argmax(1))/BATCH_SIZE
            run.log({"val_loss": output["loss"].detach().cpu().item(),
                    "val_acc": accuracy.item()})
        except RuntimeError:
            pass

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
        try:
            output = model(batch[0].float(), batch[1], batch[2], detach_transformer=True)
        except RuntimeError:
            optimizer.zero_grad()
            continue

        # backprop
        try:
            output["loss"].backward()
        except RuntimeError:
            breakpoint()
        optimizer.step()
        optimizer.zero_grad()

        # logging
        run.log({"loss": output["loss"].detach().cpu().item()})

    # scheduler.step()

# model.eval()

# # we track logits and labels and count them
# # finally together eventually
# logits = np.empty((0,3))
# labels = np.empty((0,3))
# current_targets = np.empty((0,3))

# print("Validating...")

# try:
#     # validation is large, so we do batches
#     for i in tqdm(iter(validation_loader)):
#         batch = [j.to(DEVICE) for j in i]
#         output = model(batch[0].float(), batch[1], batch[2])

#         # append to talley
#         logits = np.append(logits, output["logits"].detach().cpu().numpy(), 0)
#         labels = np.append(labels, i[2].numpy(), 0)

#         if TASK == "future":
#             # current targets used to generate comparative graph
#             current_target = F.one_hot((i[0][:,-1]*30).to(int), num_classes=3)
#             current_targets = np.append(current_targets, current_target, 0)

#         torch.cuda.empty_cache()
# except:
#     breakpoint()

# try:
#     prec_recc, roc, cm, acc = tensor_metrics(logits, labels)
#     run.log({"val_prec_recc": prec_recc,
#              "val_confusion": cm,
#              "val_roc": roc,
#              "val_acc": acc})
#     if TASK == "future":
#         (prec_recc_c, roc_c, cm_c, acc_c), (prec_recc_uc, roc_uc, cm_uc, acc_uc) = future_metrics(logits, labels,
#                                                                                                   current_targets)

#         run.log({"val_prec_recc_changed": prec_recc_c,
#                  "val_confusion_changed": cm_c,
#                  "val_roc_changed": roc_c,
#                  "val_acc_changed": acc_c})

#         run.log({"val_prec_recc_unchanged": prec_recc_uc,
#                  "val_confusion_unchanged": cm_uc,
#                  "val_roc_unchanged": roc_uc,
#                  "val_acc_unchanged": acc_uc})



#     # model.train()
# except ValueError:
#     breakpoint()

# # Saving
print("Saving model...")
os.mkdir(f"./models/{run.name}")
torch.save(model.state_dict(), f"./models/{run.name}/model.save")
torch.save(optimizer, f"./models/{run.name}/optimizer.save")

breakpoint()
