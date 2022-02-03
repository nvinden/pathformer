from dataset import *
from model import PathFormer, TokenToPosition
from process import *
from graphics import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

import time
import argparse
import os
import random
import json

np.random.seed(42)
torch.manual_seed(1609)
random.seed(56)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(boot_data, run_name):
    #loading the config file
    config_path = os.path.join("configs", boot_data['conf'] + ".json")
    IMAGE_EMBEDDING_CONFIG, DATASET_CONFIG, MODEL_CONFIG, TRAIN_CONFIG = load_json_config(config_path)

    #configuring datasets
    master_data_path = f"data/ttv_splits_{IMAGE_EMBEDDING_CONFIG['dataset_name']}.pt"
    if not os.path.isfile(master_data_path):
        dataset = PathformerData(IMAGE_EMBEDDING_CONFIG, DATASET_CONFIG, MODEL_CONFIG)

        ttv_dim = IMAGE_EMBEDDING_CONFIG['train_test_val']

        train_set, test_set, val_set = torch.utils.data.random_split(dataset, [ttv_dim[0], ttv_dim[1], ttv_dim[2]])

        torch.save([train_set, test_set, val_set], master_data_path)
    else:
        train_set, test_set, val_set = torch.load(master_data_path)

    train_loader = DataLoader(train_set, batch_size = MODEL_CONFIG['batch_size'], shuffle = True, drop_last = True)
    test_loader = DataLoader(test_set, batch_size = MODEL_CONFIG['batch_size'], shuffle = True, drop_last = True)
    val_loader = DataLoader(val_set, batch_size = MODEL_CONFIG['batch_size'], shuffle = True, drop_last = True)

    #creating model and system surrounding model
    if os.path.isfile(boot_data['path']) and boot_data['r'] == False:
        print(f"Loading from {run_name}...")
        curr_epoch, model, optim, scheduler, log_list = load_data(run_name, TRAIN_CONFIG, MODEL_CONFIG, IMAGE_EMBEDDING_CONFIG)
        curr_epoch += 1
    else:
        print(f"Creating new run...")
        curr_epoch = 0
        model = PathFormer(MODEL_CONFIG, IMAGE_EMBEDDING_CONFIG)
        optim = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size = TRAIN_CONFIG['step_size'], gamma = TRAIN_CONFIG['step_size'])
        log_list = {}

    model = model.to(device)
    
    for epoch in range(curr_epoch, TRAIN_CONFIG['n_epochs']):
        start_time = time.time()

        epoch_total_loss = 0
        epoch_total_accuracy = 0

        analyze_condition = True#(epoch % 5 == 0 and epoch != 1) or epoch == TRAIN_CONFIG['n_epochs'] - 1

        if analyze_condition:
            analysis_list = list()

            train_stem = os.path.join("results", run_name, "test")
            val_stem = os.path.join("results", run_name, "val")

            train_save_list_name = os.path.join(train_stem, f"{str(epoch).zfill(3)}.json")
            val_save_list_name = os.path.join(val_stem, f"{str(epoch).zfill(3)}.json")

            csv_filename = os.path.join("results", run_name, "results.csv")

            if not os.path.isdir(train_stem):
                os.makedirs(train_stem)

            if not os.path.isdir(val_stem):
                os.makedirs(val_stem)

        for idx, data in enumerate(train_loader):
            model.train()

            stim = data['stimuli']
            img_emb = data['image_embedding']
            seq_patch = data['sequence_patch']
            img_emb_lowres = data['lowres_stimuli']
            sequence = data['sequence']
            
            #tgt: [32, 25, 15]
            tgt = torch.clone(seq_patch)
            tgt.requires_grad = False

            if IMAGE_EMBEDDING_CONFIG['blur'] == False:
                curr_epoch_loss, curr_epoch_accuracy, batch_analyze = _train_batch(model, seq_patch, img_emb, tgt,  optim, analyze_condition, IMAGE_EMBEDDING_CONFIG)
                epoch_total_loss += curr_epoch_loss
                epoch_total_accuracy += curr_epoch_accuracy
            elif IMAGE_EMBEDDING_CONFIG['blur'] == True:
                curr_epoch_loss, curr_epoch_accuracy = _train_batch_blur(model, seq_patch, img_emb, img_emb_lowres, tgt,  optim, scheduler, boot_data)
                epoch_total_loss += curr_epoch_loss
                epoch_total_accuracy += curr_epoch_accuracy

            if analyze_condition:
                analysis_list = analysis_list + batch_analyze

            print(".", end = "")

            if idx == 2:
                break

        if analyze_condition:
            avg_dtw, avg_eye = calculate_avg_scores(analysis_list)

        epoch_total_loss /= len(train_loader)
        epoch_total_accuracy /= len(train_loader)

        if analyze_condition:
            analysis_list_val = validate(model, val_loader, boot_data, IMAGE_EMBEDDING_CONFIG, analyze_condition = analyze_condition)
            avg_dtw_val, avg_eye_val = calculate_avg_scores(analysis_list_val)

            with open(train_save_list_name, "w") as outfile:
                json_object = json.dumps(analysis_list, indent = 4)
                outfile.write(json_object)

            with open(val_save_list_name, "w") as outfile:
                json_object = json.dumps(analysis_list_val, indent = 4)
                outfile.write(json_object)

            with open(csv_filename,'a') as outfile:
                out_str = f"{epoch},{avg_dtw},{avg_eye},{avg_dtw_val},{avg_eye_val}\n"
                outfile.write(out_str)


        print(f"\nEpoch {epoch}:")
        print(f"         loss: {optim.defaults['lr']}")
        print(f"      avg dtw: {avg_dtw}")
        print(f"      avg eye: {avg_eye}")
        print(f"  val avg dtw: {avg_dtw_val}")
        print(f"  val avg eye: {avg_eye_val}")
        print(f" TIME: {time.time() - start_time}")

        scheduler.step()

        save_data(run_name, epoch, model, scheduler, optim, log_list)

    print("Training complete")

def _train_batch_blur(model, seq_patch, img_emb, img_emb_blur, target, optim, analyze_condition):
    total_loss = 0
    total_accuracy = 0
    for i in range(seq_patch.shape[0]):
        optim.zero_grad()

        curr_seq_patch = seq_patch[:, i, :]
        curr_target = target[:, i, :]

        curr_view = torch.clone(img_emb_blur)

        for j in range(curr_seq_patch.shape[1]):
            result, _ = model(curr_seq_patch, curr_view)

            loss = calculate_loss(result, curr_target, model)
            loss.backward()

            optim.step()

            total_loss += loss
        
            total_accuracy += percent_on_correct(result, curr_target, False)

            for k in range(curr_seq_patch.shape[0]):
                if curr_seq_patch[k, j] == model.END:
                    curr_seq_patch = np.delete(curr_seq_patch, )

def _train_batch(model, seq_patch, img_emb, target, optim, analyze_condition, IMAGE_EMBEDDING_CONFIG):
    total_loss = 0
    total_accuracy = 0

    if analyze_condition:
        analyze_batch_list = [{} for i in range(seq_patch.shape[0])]

    for i in range(seq_patch.shape[1]):
        optim.zero_grad()

        curr_seq_patch = seq_patch[:, i, :]
        curr_target = target[:, i, :]

        curr_seq_patch = curr_seq_patch.to(device)
        curr_target = curr_seq_patch.to(device)

        result, _ = model(curr_seq_patch, img_emb)

        loss = calculate_loss(result, curr_target, model)
        loss.backward()

        optim.step()

        total_loss += loss

        if analyze_condition:
            increment_analyze_batch_list(analyze_batch_list, result, curr_target, IMAGE_EMBEDDING_CONFIG)
    
        total_accuracy += percent_on_correct(result, curr_target, False)

    if analyze_condition:
        finalize_analyze_batch_list(analyze_batch_list)
    else:
        analyze_batch_list = None

    return total_loss / seq_patch.shape[1], total_accuracy / seq_patch.shape[1], analyze_batch_list

def train_on_tokens(boot_data):
    TOKEN_EPOCHS = 100

    #loading the config file
    spec = importlib.util.spec_from_file_location("*", boot_data['conf'])
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)

    IMAGE_EMBEDDING_CONFIG = foo.IMAGE_EMBEDDING_CONFIG
    DATASET_CONFIG = foo.DATASET_CONFIG
    MODEL_CONFIG = foo.MODEL_CONFIG
    TRAIN_CONFIG = foo.TRAIN_CONFIG

    train_on_tokens_path = os.path.join(DATASET_CONFIG['data_path'], "PathFormerTokens", "model")

    #configuring datasets
    dataset = PathformerTokenData(IMAGE_EMBEDDING_CONFIG, DATASET_CONFIG, MODEL_CONFIG)

    ttv_dim = IMAGE_EMBEDDING_CONFIG['train_test_val']

    loader = DataLoader(dataset, batch_size = 32, shuffle = True, drop_last = True)

    criterion = nn.MSELoss()

    if not os.path.isfile(train_on_tokens_path):
        model = TokenToPosition()
        model.train()
        
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        epoch_start = 0

        log_list = list()
    else:
        data = torch.load(train_on_tokens_path)

        epoch_start = data['epoch']
        model = data['model_state_dict']
        optimizer = data["optimizer_state_dict"]
        log_list = data["log_list"]

    for epoch in range(epoch_start, TOKEN_EPOCHS):
        start_time = time.time()

        batch_loss_list = list()

        for batch_no, data in enumerate(loader):
            optimizer.zero_grad()

            seq_pos = data['seq_pos']
            img = data['img']

            out, _ = model(img)

            loss = criterion(out, seq_pos)
            loss.backward()

            optimizer.step()

            batch_loss_list.append(loss)

            print(f"Batch {batch_no}")

        curr_loss = sum(batch_loss_list) / len(batch_loss_list)
        log_list.append(curr_loss)

        print(f"Epoch {epoch}:")
        print(f"           lr: {optim.defaults['lr']}")
        print(f"         Loss: {curr_loss}")
        print(f" TIME: {time.time() - start_time}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'log_list': log_list
        }, train_on_tokens_path)
         
def main():
    # on_self: training encoder and decoder on the same path. Good for first epoch
    # on_pic: training encoder on one path and decoder on another.
    # full: trains on

    parser = argparse.ArgumentParser(description='Pathformer: Scanpaths can attend too')
    parser.add_argument('--path', help='path for save/load of data')
    parser.add_argument('--conf', help='path for the config file')
    parser.add_argument('-v', help='verbose mode', action='store_true')
    parser.add_argument('-r', help='re-download dataset', action='store_true')
    parser.add_argument('-t', help='train the vgg19 coder', action='store_true')

    args = parser.parse_args()
    boot_info = vars(args)

    if boot_info['t']:
        train_on_tokens(boot_info)
    else:
        train(boot_info, "test_run")

if __name__ == '__main__':
    main()