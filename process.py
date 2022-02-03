
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import PathFormer
from dataset import get_data_from_batch_number, create_target_sequence

from saliency.metrics import eyenalysis, DTW

import math
import os

import numpy as np

import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_json_config(path):
    with open(path) as f:
        data = json.load(f)

    return data["IMAGE_EMBEDDING_CONFIG"], data["DATASET_CONFIG"], data["MODEL_CONFIG"], data["TRAIN_CONFIG"]

def save_data(path, epoch, model, scheduler, optimizer, log_list):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'log_list': log_list
    }, path)

def load_data(path, TRAIN_CONFIG, MODEL_CONFIG, IMAGE_EMBEDDING_CONFIG):
    data = torch.load(path)

    epoch = data['epoch']
    log_list = data['log_list']

    model = PathFormer(MODEL_CONFIG, IMAGE_EMBEDDING_CONFIG)
    model.load_state_dict(data['model_state_dict'])

    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['lr'])
    optimizer.load_state_dict(data['optimizer_state_dict'])
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, TRAIN_CONFIG['step_size'], gamma=TRAIN_CONFIG['gamma'])
    scheduler.load_state_dict(data['scheduler_state_dict'])

    return epoch, model, optimizer, scheduler, log_list

def create_distance_dictionary(model, IMAGE_EMBEDDING_CONFIG, path = None):
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

def spatio_mse(predicted, target, model, IMAGE_EMBEDDING_CONFIG):
    pred = torch.clone(predicted).to(device)
    tgt = torch.clone(target).to(device)

    dist_dict = create_distance_dictionary(model, IMAGE_EMBEDDING_CONFIG, path = None)

    s_mse = 0

    for curr_pred, curr_targ in zip(pred, tgt):
        for tgt_num, curr_row in zip(curr_targ, curr_pred):
            if tgt_num == model.END:
                break

            curr_row = curr_row[:model.n_patches]
            curr_row = F.softmax(curr_row, dim = 0)
            
            torch_dict = torch.from_numpy(dist_dict[tgt_num, :]).to(device)
            torch_dict.requires_grad = False

            ew_mult = torch_dict * curr_row
            s_mse += ew_mult.sum()

    s_mse /= (tgt.shape[0] * tgt.shape[1])

    return s_mse

def point_type_crossentropy_loss(predicted, target, model):
    cel = nn.CrossEntropyLoss()

    n_out_dims = model.n_tokens - model.n_patches + 1

    new_pred = torch.zeros([predicted.shape[0], predicted.shape[1], n_out_dims], device = device)
    new_pred[:, :, 1:] = predicted[:, :, -n_out_dims + 1:]
    new_pred[:, :, 0] = 1 - new_pred.sum(axis = 2)

    tgt_cpy = torch.clone(target).to(device = device)
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
    cel = nn.NLLLoss(weight = model.ds_token_weights, ignore_index = model.NONE)

    pred = predicted.permute([0, 2, 1])
    pred = torch.log(pred)

    out = cel(pred, target)

    '''
    out = 0

    for pred, tgt in zip(predicted, target):
        curr_out = 0
        if tgt[-1] != model.NONE:
            index = len(tgt) - 1
        else:
            index = ((tgt == model.NONE).nonzero()[0])

        pred = torch.log(pred)

        curr_out = cel(pred[:index], tgt[:index])
        if isinstance(index, int):
            curr_out /= index
        else:
            curr_out /= index.item()

        out += curr_out
    out /= model.batch_size

    '''

    return out

def calculate_loss(predicted, target, model):
    #predicted (batch, n_tokens, n_patches)
    #target (batch, n_patches)

    weights = [1.0, 0.0]#TRAIN_CONFIG['loss_weights']

    pt_cse = cross_entropy_loss(predicted, target, model)

    s_mse = 0.0#spatio_mse(predicted, target, model)

    return weights[0] * pt_cse + weights[1] * s_mse

def percent_on_correct(result, target, verbose):
    #result: [32, 25, 52]
    #target: [32, 25]
    avg_correctness = 0
    div = 0

    for res_batch, tgt_batch in zip(result, target):
        for res_point, tgt_point in zip(res_batch, tgt_batch):
            if tgt_point == 51:
                break
            div += 1
            avg_correctness += res_point[tgt_point]


    avg_correctness /= div

    return avg_correctness

def calculate_avg_scores(analysis_list):
    total_dtw = 0.0
    total_eye = 0.0

    for batch in analysis_list:
        total_dtw += batch["dtw_spp"]
        total_eye += batch["eye_spp"]

    avg_dtw = total_dtw / len(analysis_list)
    avg_eye = total_eye / len(analysis_list)

    return avg_dtw, avg_eye

def finalize_analyze_batch_list(analyze_batch_list):
    for batch in analyze_batch_list:
        batch["dtw_spp"] = min(batch["dtw"])
        batch["eye_spp"] = min(batch["eye"])

def increment_analyze_batch_list(analyze_batch_list, result, curr_target, IMAGE_EMBEDDING_CONFIG):
    #tokens: List of single tokens that are used in transformer [19, 18, 32, 43, 51, 51, 51, 51]
    #xy: x-y coordinates in the range (0 to 1)
    #token-xy: List of tokens used in dtw in the form [(x_tok1, y_tok1), (x_tok2, y_tok2), ...]
    token_results = result_to_tokens(result)
    tokenxy_results = tokens_to_tokenxy(token_results, IMAGE_EMBEDDING_CONFIG)
    xy_results = tokens_to_xy(token_results, IMAGE_EMBEDDING_CONFIG)

    tokenxy_target = tokens_to_tokenxy(curr_target, IMAGE_EMBEDDING_CONFIG)
    xy_target = tokens_to_xy(curr_target, IMAGE_EMBEDDING_CONFIG)

    for batch_no, (tokenxy_series_results, xy_series_results, tokenxy_series_target, xy_series_target) in enumerate(zip(tokenxy_results, xy_results, tokenxy_target, xy_target)):
        curr_batch = analyze_batch_list[batch_no]
        dtw_distance = DTW(tokenxy_series_results, tokenxy_series_target)
        eye_distance = eyenalysis(xy_series_results, xy_series_target)

        if "dtw" not in curr_batch:
            curr_batch["dtw"] = list()
        if "eye" not in curr_batch:
            curr_batch["eye"] = list()

        curr_batch["dtw"].append(dtw_distance)
        curr_batch["eye"].append(eye_distance)

def increment_analyze_batch_list_val(analyze_batch_list, token_results, curr_target, IMAGE_EMBEDDING_CONFIG):
    #tokens: List of single tokens that are used in transformer [19, 18, 32, 43, 51, 51, 51, 51]
    #xy: x-y coordinates in the range (0 to 1)
    #token-xy: List of tokens used in dtw in the form [(x_tok1, y_tok1), (x_tok2, y_tok2), ...]
    tokenxy_results = tokens_to_tokenxy(token_results, IMAGE_EMBEDDING_CONFIG)
    xy_results = tokens_to_xy(token_results, IMAGE_EMBEDDING_CONFIG)

    tokenxy_target = tokens_to_tokenxy(curr_target, IMAGE_EMBEDDING_CONFIG)
    xy_target = tokens_to_xy(curr_target, IMAGE_EMBEDDING_CONFIG)

    for batch_no, (tokenxy_series_results, xy_series_results, tokenxy_series_target, xy_series_target) in enumerate(zip(tokenxy_results, xy_results, tokenxy_target, xy_target)):
        curr_batch = analyze_batch_list[batch_no]
        dtw_distance = DTW(tokenxy_series_results, tokenxy_series_target)
        eye_distance = eyenalysis(xy_series_results, xy_series_target)

        if "dtw" not in curr_batch:
            curr_batch["dtw"] = list()
        if "eye" not in curr_batch:
            curr_batch["eye"] = list()

        curr_batch["dtw"].append(dtw_distance)
        curr_batch["eye"].append(eye_distance)
        
def tokens_to_xy(tokens, config):
    height_split = config["image_splits"][0]
    width_split = config["image_splits"][1]

    out = list()

    for batch in tokens:
        xy_series = list()

        for token in batch:
            token = token.item()

            if token >= height_split * width_split:
                break

            token_y = token // width_split
            token_x = token % width_split

            y = (float(token_y) + 0.5) / height_split
            x = (float(token_x) + 0.5) / width_split

            xy_series.append([x, y])

        out.append(xy_series)

    return out

def tokens_to_tokenxy(tokens, config):
    height_split = config["image_splits"][0]
    width_split = config["image_splits"][1]

    out = list()

    for batch in tokens:
        tokenxy_series = list()

        for token in batch:
            token = token.item()

            if token >= height_split * width_split:
                break

            token_y = token // width_split
            token_x = token % width_split

            tokenxy_series.append([token_x, token_y])

        out.append(tokenxy_series)

    return out

def result_to_tokens(result):
    assert len(result.shape) == 2 or len(result.shape) == 3
    if len(result.shape) == 2:
        return torch.argmax(result, dim = 1)
    else:
        return torch.argmax(result, dim = 2)

def validate(model, val_loader, boot_data, IMAGE_EMBEDDING_CONFIG, analyze_condition):
    loss_count = 0
    accuracy_count = 0

    model.eval()

    if analyze_condition:
        analysis_list = list()

    for idx, data in enumerate(val_loader):
        model.eval()

        stim = data['stimuli']
        img_emb = data['image_embedding']
        seq_patch = data['sequence_patch']

        target = torch.clone(seq_patch).to(device = device)

        for i in range(seq_patch.shape[1]):

            if analyze_condition:
                analyze_batch_list = [{} for i in range(seq_patch.shape[0])]

            curr_seq_patch = seq_patch[:, i, :]
            curr_target = target[:, i, :]

            result = model(curr_seq_patch, img_emb)

            if analyze_condition:
                increment_analyze_batch_list_val(analyze_batch_list, result, curr_target, IMAGE_EMBEDDING_CONFIG)

        if analyze_condition:
            analysis_list = analysis_list + analyze_batch_list

    if analyze_condition:
        finalize_analyze_batch_list(analyze_batch_list)
    else:
        analyze_batch_list = None
    
    loss_count /= len(val_loader)
    accuracy_count /= len(val_loader)

    return analyze_batch_list

def add_to_log_list(log_list, train_mode, epoch_total_loss, epoch_total_accuracy, val_loss, val_accuracy):
    if train_mode in log_list:
        log_list[train_mode].append([epoch_total_loss, epoch_total_accuracy, val_loss, val_accuracy])
    else:
        log_list[train_mode] = [epoch_total_loss, epoch_total_accuracy, val_loss, val_accuracy]




        
