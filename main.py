from saliency import dataset
from config import *
from dataset import ImageEmbeddings, create_batches, create_target_sequence, get_data_from_batch_number
from model import PathFormer
from process import calculate_loss, print_percent_on_correct, load_data, save_data

import torch
import torch.nn as nn

import numpy as np

import time
import argparse
import importlib.util
import os

from torch.utils.data import Dataset, DataLoader

def train(boot_data):
    #loading the config file
    spec = importlib.util.spec_from_file_location("*", boot_data['conf'])
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)

    global IMAGE_EMBEDDING_CONFIG
    global DATASET_CONFIG
    global MODEL_CONFIG
    global TRAIN_CONFIG

    IMAGE_EMBEDDING_CONFIG = foo.IMAGE_EMBEDDING_CONFIG
    DATASET_CONFIG = foo.DATASET_CONFIG
    MODEL_CONFIG = foo.MODEL_CONFIG
    TRAIN_CONFIG = foo.TRAIN_CONFIG

    #creating model and system surrounding model
    if os.path.isfile(boot_data['path']):
        curr_epoch, train_method, model, optim, scheduler, loss = load_data(boot_data['path'])
    else:
        curr_epoch = 0
        train_method = "on_self"
        model = PathFormer(MODEL_CONFIG, IMAGE_EMBEDDING_CONFIG, train_method)
        optim = torch.optim.SGD(model.parameters(), lr=TRAIN_CONFIG['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optim, 1.0, gamma=0.95)

    train_max_range = (IMAGE_EMBEDDING_CONFIG['train_idx'][1] - IMAGE_EMBEDDING_CONFIG['train_idx'][0]) // MODEL_CONFIG['batch_size']
    test_max_range = (IMAGE_EMBEDDING_CONFIG['test_idx'][1] - IMAGE_EMBEDDING_CONFIG['test_idx'][0]) // MODEL_CONFIG['batch_size']
    val_max_range = (IMAGE_EMBEDDING_CONFIG['val_idx'][1] - IMAGE_EMBEDDING_CONFIG['val_idx'][0]) // MODEL_CONFIG['batch_size']
        
    model.train()

    start_time = time.time()

    for epoch in range(curr_epoch, TRAIN_CONFIG['n_epochs']):
        for i_batch in range(train_max_range):
            (seq, stim, img_emb, seq_patch) = get_data_from_batch_number(i_batch, IMAGE_EMBEDDING_CONFIG, boot_data['r'], "train")

            tgt = create_target_sequence(seq_patch, model)
            tgt.requires_grad = False

            if train_method == "on_self":
                loss = _train_on_self(model, seq_patch, img_emb, tgt,  optim, scheduler, boot_data)
            elif train_method == "on_pic":
                pass

        print(f"epoch {epoch} {train_method} loss: {loss}")

        save_data(boot_data['path'], epoch, model, scheduler, train_method, optim, loss)

def _train_on_self(model, seq_patch, img_emb, target, optim, scheduler, boot_data):
    total_loss = 0
    for i in range(seq_patch.shape[1]):
        optim.zero_grad()

        curr_seq_patch = seq_patch[:, i]
        curr_target = target[:, :, i]

        result = model(curr_seq_patch, img_emb)

        loss = calculate_loss(result, curr_target, model)
        loss.backward()

        optim.step()

        total_loss += loss

        print_percent_on_correct(result, curr_target, boot_data['v'])

    return total_loss / seq_patch.shape[1]

def main():
    # on_self: training encoder and decoder on the same path. Good for first epoch
    # on_pic: training encoder on one path and decoder on another.

    parser = argparse.ArgumentParser(description='Pathformer: Scanpaths can attend too')
    parser.add_argument('--path', help='path for save/load of data')
    parser.add_argument('--conf', help='path for the config file')
    parser.add_argument('-v', help='verbose mode', action='store_true')
    parser.add_argument('-r', help='re-download dataset', action='store_true')

    args = parser.parse_args()
    boot_info = vars(args)

    train(boot_info)

if __name__ == '__main__':
    main()