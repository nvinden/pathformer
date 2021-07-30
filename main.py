from saliency import dataset
from config import *
from dataset import ImageEmbeddings, create_batches, create_target_sequence
from model import PathFormer
from process import calculate_loss, print_percent_on_correct

import torch
import torch.nn as nn

import numpy as np

import time

from torch.utils.data import Dataset, DataLoader

TF = TRAIN_CONFIG

def train(train_method):
    ds = ImageEmbeddings(IMAGE_EMBEDDING_CONFIG, False)

    batch_size = MODEL_CONFIG['batch_size']
    seq_batches = create_batches(ds.seq, batch_size)
    stim_batches = create_batches(ds.stim, batch_size)
    img_emb_batches = create_batches(ds.img_emb, batch_size)
    seq_patch_batches = create_batches(ds.pos_emb, batch_size)

    model = PathFormer(MODEL_CONFIG, IMAGE_EMBEDDING_CONFIG, train_method)
    model.train()

    start_time = time.time()

    optim = torch.optim.SGD(model.parameters(), lr=TF['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 1.0, gamma=0.95)

    for epoch in range(TF['n_epochs']):
        for i_batch, data in enumerate(zip(seq_batches, stim_batches, img_emb_batches, seq_patch_batches)):
            (seq, stim, img_emb, seq_patch) = data

            tgt = create_target_sequence(seq_patch, model)
            tgt.requires_grad = False

            if train_method == "on_self":
                _train_on_self(model, seq_patch, img_emb, tgt,  optim, scheduler)
            elif train_method == "on_pic":
                pass

            result = model(seq_patch[:, 0], seq_patch[:, 0], img_emb)

def _train_on_self(model, seq_patch, img_emb, target, optim, scheduler):
    for i in range(seq_patch.shape[1]):
        optim.zero_grad()

        curr_seq_patch = seq_patch[:, i]
        curr_target = target[:, :, i]

        result = model(curr_seq_patch, img_emb)

        loss = calculate_loss(result, curr_target, model)
        loss.backward()

        optim.step()

        print(loss)
        print(result[0][0])
        print_percent_on_correct(result, curr_target)
        
        

def main():
    # on_self: training encoder and decoder on the same path. Good for first epoch
    # on_pic: training encoder on one path and decoder on another.
    train_method = "on_self"

    train(train_method)


if __name__ == '__main__':
    main()