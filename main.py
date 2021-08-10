from saliency import dataset
from config import *
from dataset import *
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

    #configuring datasets
    dataset = PathformerData(IMAGE_EMBEDDING_CONFIG, DATASET_CONFIG, MODEL_CONFIG)

    ttv_dim = IMAGE_EMBEDDING_CONFIG['train_test_val']

    train_set, test_set, val_set = torch.utils.data.random_split(dataset, [ttv_dim[0], ttv_dim[1], ttv_dim[2]])

    train_loader = DataLoader(train_set, batch_size = MODEL_CONFIG['batch_size'], shuffle = True, drop_last = True)
    test_loader = DataLoader(test_set, batch_size = MODEL_CONFIG['batch_size'], shuffle = True, drop_last = True)
    val_loader = DataLoader(val_set, batch_size = MODEL_CONFIG['batch_size'], shuffle = True, drop_last = True)

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
    

    while CURR_TRAIN_METHOD != "end":
        model.train_method = CURR_TRAIN_METHOD
        for epoch in range(curr_epoch, CURR_TRAIN_CONFIG['n_epochs']):
            start_time = time.time()

            epoch_total_loss = 0
            epoch_total_accuracy = 0

            for data in train_loader:
                model.train()

                stim = data['stimuli']
                img_emb = data['image_embedding']
                seq_patch = data['sequence_patch']
                
                #tgt: [32, 25, 15]
                tgt = create_target_sequence(seq_patch, model)
                tgt.requires_grad = False

                if CURR_TRAIN_METHOD == "on_self":
                    curr_epoch_loss, curr_epoch_accuracy = _train_on_self(model, seq_patch, img_emb, tgt,  optim, scheduler, boot_data)
                    epoch_total_loss += curr_epoch_loss
                    epoch_total_accuracy += curr_epoch_accuracy
                elif CURR_TRAIN_METHOD == "on_pic":
                    print("ON PIC EPOCH")
                elif CURR_TRAIN_METHOD == "full":
                    print("FULL EPOCH")

                print(epoch_total_loss)
                print(epoch_total_accuracy)

            epoch_total_loss /= len(train_loader)
            epoch_total_accuracy /= len(train_loader)

            val_loss, val_accuracy = validate(model, val_loader, boot_data)

            add_to_log_list(log_list, CURR_TRAIN_METHOD, epoch_total_loss, epoch_total_accuracy, val_loss, val_accuracy)

            print(f"Epoch {epoch}: on {CURR_TRAIN_METHOD}")
            print(f"         Loss: {epoch_total_loss}")
            print(f"     Accuracy: {epoch_total_accuracy}")
            print(f"     Val Loss: {val_loss}")
            print(f" Val Accuracy: {val_accuracy}")
            print(f" TIME: {time.time() - start_time}")

            save_data(boot_data['path'], epoch, model, scheduler, CURR_TRAIN_METHOD, optim, log_list)
        
        #switching to next phase
        if CURR_TRAIN_METHOD == "on_self":
            if "on_pic" in TRAIN_CONFIG:
                CURR_TRAIN_METHOD = "on_pic"
                CURR_TRAIN_CONFIG = TRAIN_CONFIG["on_pic"]
            elif "full" in TRAIN_CONFIG:
                CURR_TRAIN_METHOD = "full"
                CURR_TRAIN_CONFIG = TRAIN_CONFIG["full"]
            else:
                CURR_TRAIN_METHOD = "end"
        elif CURR_TRAIN_METHOD == "on_pic":
            if "full" in TRAIN_CONFIG:
                CURR_TRAIN_METHOD = "full"
                CURR_TRAIN_CONFIG = TRAIN_CONFIG["full"]
            else:
                CURR_TRAIN_METHOD = "end"
        elif CURR_TRAIN_METHOD == "full":
            CURR_TRAIN_METHOD = "end"
        else:
            raise ValueError("Incorrect train method type")

def _train_on_self(model, seq_patch, img_emb, target, optim, scheduler, boot_data):
    total_loss = 0
    total_accuracy = 0
    for i in range(seq_patch.shape[1]):
        optim.zero_grad()

        curr_seq_patch = seq_patch[:, i, :]
        curr_target = target[:, i, :]

        result = model(curr_seq_patch, img_emb)

        loss = calculate_loss(result, curr_target, model)
        loss.backward()

        optim.step()

        total_loss += loss
    
        total_accuracy += percent_on_correct(result, curr_target, False)

    return total_loss / seq_patch.shape[1], total_accuracy / seq_patch.shape[1]

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