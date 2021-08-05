from saliency import dataset
from config import *
from dataset import ImageEmbeddings, create_batches, create_target_sequence, get_data_from_batch_number
from model import PathFormer
from process import *

import torch
import torch.nn as nn

import numpy as np

import time
import argparse
import importlib.util
import os

from torch.utils.data import Dataset, DataLoader

global IMAGE_EMBEDDING_CONFIG
global DATASET_CONFIG
global MODEL_CONFIG
global TRAIN_CONFIG
global CURR_TRAIN_CONFIG
global CURR_TRAIN_METHOD

def train(boot_data):
    #loading the config file
    spec = importlib.util.spec_from_file_location("*", boot_data['conf'])
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)

    IMAGE_EMBEDDING_CONFIG = foo.IMAGE_EMBEDDING_CONFIG
    DATASET_CONFIG = foo.DATASET_CONFIG
    MODEL_CONFIG = foo.MODEL_CONFIG
    TRAIN_CONFIG = foo.TRAIN_CONFIG

    #creating model and system surrounding model
    if os.path.isfile(boot_data['path']) and boot_data['r'] == False:
        curr_epoch, CURR_TRAIN_METHOD, model, optim, scheduler, log_list = load_data(boot_data['path'])
    else:
        curr_epoch = 0
        CURR_TRAIN_METHOD = "on_self"
        model = PathFormer(MODEL_CONFIG, IMAGE_EMBEDDING_CONFIG, CURR_TRAIN_METHOD)
        optim = torch.optim.SGD(model.parameters(), lr=TRAIN_CONFIG['on_self']['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optim, 1.0, gamma=0.95)
        log_list = {}

    CURR_TRAIN_CONFIG = TRAIN_CONFIG[CURR_TRAIN_METHOD]

    train_max_range = (IMAGE_EMBEDDING_CONFIG['train_idx'][1] - IMAGE_EMBEDDING_CONFIG['train_idx'][0]) // MODEL_CONFIG['batch_size']
    test_max_range = (IMAGE_EMBEDDING_CONFIG['test_idx'][1] - IMAGE_EMBEDDING_CONFIG['test_idx'][0]) // MODEL_CONFIG['batch_size']
    val_max_range = (IMAGE_EMBEDDING_CONFIG['val_idx'][1] - IMAGE_EMBEDDING_CONFIG['val_idx'][0]) // MODEL_CONFIG['batch_size']
        
    model.train()

    start_time = time.time()

    while CURR_TRAIN_METHOD != "end":
        model.train_method = CURR_TRAIN_METHOD
        for epoch in range(curr_epoch, CURR_TRAIN_CONFIG['n_epochs']):
            epoch_total_loss = 0
            epoch_total_accuracy = 0

            for i_batch in range(train_max_range):
                val_loss, val_accuracy = validate(model, val_max_range, boot_data)

                model.train()
                (seq, stim, img_emb, seq_patch) = get_data_from_batch_number(i_batch, IMAGE_EMBEDDING_CONFIG, boot_data['r'], "train")

                tgt = create_target_sequence(seq_patch, model)
                tgt.requires_grad = False

                if CURR_TRAIN_METHOD == "on_self":
                    epoch_total_loss += _train_on_self(model, seq_patch, img_emb, tgt,  optim, scheduler, boot_data)
                elif CURR_TRAIN_METHOD == "on_pic":
                    print("ON PIC EPOCH")
                elif train_method == "full":
                    print("FULL EPOCH")

            epoch_total_loss /= train_max_range
            epoch_total_accuracy /= train_max_range

            val_loss, val_accuracy = validate(model, val_max_range, boot_data)

            add_to_log_list(log_list, CURR_TRAIN_METHOD, epoch_total_loss, epoch_total_accuracy, val_loss, val_accuracy)

            print(f"Epoch {epoch}: on {CURR_TRAIN_METHOD}")
            print(f"        Loss: {epoch_total_loss}")
            print(f"    Accuracy: {epoch_total_accuracy}")
            print(f"    Val Loss: {val_loss}")
            print(f"Val Accuracy: {val_accuracy}")

            save_data(boot_data['path'], epoch, model, scheduler, CURR_TRAIN_METHOD, optim, log_list)
        
        #switching to next phase
        if train_method == "on_self":
            if "on_pic" in TRAIN_CONFIG:
                train_method = "on_pic"
                CURR_TRAIN_CONFIG = TRAIN_CONFIG["on_pic"]
            elif "full" in TRAIN_CONFIG:
                train_method = "full"
                CURR_TRAIN_CONFIG = TRAIN_CONFIG["full"]
            else:
                train_method = "end"
        elif train_method == "on_pic":
            if "full" in TRAIN_CONFIG:
                train_method = "full"
                CURR_TRAIN_CONFIG = TRAIN_CONFIG["full"]
            else:
                train_method = "end"
        elif train_method == "full":
            train_method = "end"
        else:
            raise ValueError("Incorrect train method type")

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

    return total_loss / seq_patch.shape[1]

def main():
    # on_self: training encoder and decoder on the same path. Good for first epoch
    # on_pic: training encoder on one path and decoder on another.
    # full: trains on

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