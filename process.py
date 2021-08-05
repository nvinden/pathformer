from config import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import PathFormer
from config import *
from dataset import get_data_from_batch_number, create_target_sequence

import math
import os

import numpy as np

def save_data(path, epoch, model, scheduler, train_method, optimizer, log_list):
    torch.save({
            'epoch': epoch,
            'train_method': train_method,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'log_list': log_list
    }, path)

def load_data(path):
    data = torch.load(path)

    epoch = data['epoch']
    train_method = data['train_method']
    log_list = data['log_list']

    model = PathFormer(MODEL_CONFIG, IMAGE_EMBEDDING_CONFIG, train_method)
    model.load_state_dict(data['model_state_dict'])

    optim = torch.optim.SGD(model.parameters(), lr=TRAIN_CONFIG[train_method]['lr'])
    optim.load_state_dict(data['optimizer_state_dict'])

    scheduler = torch.optim.lr_scheduler.StepLR(optim, 1.0, gamma=0.95)
    scheduler.load_state_dict(data['scheduler_state_dict'])

    return epoch, train_method, model, optim, scheduler, log_list

def create_distance_dictionary(model, path = None):
    if path is not None and os.path.isfile(path):
        dist_dict = np.load(path)
    else:
        height_splits = IMAGE_EMBEDDING_CONFIG['image_splits'][0]
        width_splits = IMAGE_EMBEDDING_CONFIG['image_splits'][1]

        dist_dict = np.zeros([model.n_patches, model.n_patches], dtype = np.float32)

        for tgt in range(model.n_patches):
            for pred in range(model.n_patches):
                tgt_height = tgt // width_splits
                tgt_width = tgt % width_splits

                pred_height = pred // width_splits
                pred_width = pred % width_splits

                height_dist = np.abs(pred_height - tgt_height)
                width_dist = np.abs(pred_width - tgt_width)

                dist = np.sqrt(height_dist ** 2 + width_dist ** 2)

                dist_dict[pred, tgt] = dist

        if path is not None:
            np.save(path, dist_dict)

    return dist_dict


def spatio_mse(predicted, target, model):
    pred = torch.clone(predicted)
    tgt = torch.clone(target)

    dist_dict = create_distance_dictionary(model, path = None)

    s_mse = 0

    for curr_pred, curr_targ in zip(pred, tgt):
        for tgt_num, curr_row in zip(curr_targ, curr_pred):
            if tgt_num == model.END:
                break

            curr_row = curr_row[:model.n_patches]
            curr_row = F.softmax(curr_row, dim = 0)
            
            torch_dict = torch.from_numpy(dist_dict[tgt_num, :])
            torch_dict.requires_grad = False

            ew_mult = torch_dict * curr_row
            s_mse += ew_mult.sum()

    s_mse /= (tgt.shape[0] * tgt.shape[1])

    return s_mse

def point_type_crossentropy_loss(predicted, target, model):
    cel = nn.CrossEntropyLoss()

    n_out_dims = model.n_tokens - model.n_patches + 1

    new_pred = torch.zeros([predicted.shape[0], predicted.shape[1], n_out_dims])
    new_pred[:, :, 1:] = predicted[:, :, -n_out_dims + 1:]
    new_pred[:, :, 0] = 1 - new_pred.sum(axis = 2)

    tgt_cpy = torch.clone(target)
    for i, batch in enumerate(tgt_cpy):
        for j, point in enumerate(batch):
            if 0 <= point <= model.n_patches - 1:
                tgt_cpy[i, j] = 0
            else:
                tgt_cpy[i, j] = point - model.n_patches

    out = 0
    for pred, tar in zip(new_pred, tgt_cpy):
        out += cel(pred, tar)

    return out

def cross_entropy_loss(predicted, target, model):
    cel = nn.NLLLoss()

    out = 0

    for pred, tgt in zip(predicted, target):
        curr_out = 0
        index = ((tgt == model.NONE).nonzero()[0])

        pred = torch.log(pred)

        curr_out = cel(pred[:index], tgt[:index])
        curr_out /= index.item()

        out += curr_out
    out /= model.batch_size

    return out

def calculate_loss(predicted, target, model):
    #predicted (batch, n_tokens, n_patches)
    #target (batch, n_patches)

    weights = TRAIN_CONFIG[model.train_method]['loss_weights']

    pt_cse = cross_entropy_loss(predicted, target, model)

    s_mse = 0#spatio_mse(predicted, target, model)

    return weights[0] * pt_cse + weights[1] * s_mse

def percent_on_correct(result, target, verbose):
    avg_correctness = 0
    div = 0

    for res_batch, tgt_batch in zip(result, target):
        for res_viewer, tgt_viewer in zip(res_batch, tgt_batch):
            for res, tgt in zip(res_viewer, tgt_viewer):
                if tgt == 51:
                    break
                div += 1
                avg_correctness += res_viewer[tgt]
                max_idx = torch.argmax(res)
                if verbose:
                    print(tgt, max_idx, res[tgt])

    avg_correctness /= div

    return avg_correctness

def validate(model, val_max_range, boot_data):
    loss_count = 0
    accuracy_count = 0

    model.eval()
    for i_batch in range(val_max_range):
        (seq, stim, img_emb, seq_patch) = get_data_from_batch_number(i_batch, IMAGE_EMBEDDING_CONFIG, boot_data['r'], "val")

        viewer_loss_count = 0
        viewer_accuracy_count = 0

        for i in range(seq_patch.shape[-1]):
            tgt = create_target_sequence(seq_patch, model)
            tgt.requires_grad = False

            curr_seq_patch = seq_patch[:, i]
            curr_target = tgt[:, :, i]

            result = model(curr_seq_patch, img_emb)

            viewer_loss_count += calculate_loss(result, curr_target, model)
            viewer_accuracy_count += percent_on_correct(result, tgt, False)

        loss_count += viewer_loss_count / seq_patch.shape[-1]
        accuracy_count += viewer_accuracy_count / seq_patch.shape[-1]
    
    loss_count /= val_max_range
    accuracy_count /= val_max_range

    return loss_count, accuracy_count

def add_to_log_list(log_list, train_mode, epoch_total_loss, epoch_total_accuracy, val_loss, val_accuracy):
    if train_mode in log_list:
        log_list[train_mode].append([epoch_total_loss, epoch_total_accuracy, val_loss, val_accuracy])
    else:
        log_list[train_mode] = [epoch_total_loss, epoch_total_accuracy, val_loss, val_accuracy]




        
