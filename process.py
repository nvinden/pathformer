from config import *

import torch
import torch.nn as nn

import math
import os

import numpy as np


def create_distance_dictionary(model, path = None):
    if path is not None and os.path.isfile(path):
        dist_dict = np.load(path)
    else:
        height_splits = IMAGE_EMBEDDING_CONFIG['image_splits'][0]
        width_splits = IMAGE_EMBEDDING_CONFIG['image_splits'][1]

        dist_dict = np.zeros([model.n_tokens, model.n_tokens], dtype = np.float32)

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
    predicted = predicted.permute(0, 2, 1)

    dist_dict = create_distance_dictionary(model, path = None)

    s_mse = 0

    for curr_pred, curr_targ in zip(predicted, target):
        for tgt_num, curr_row in zip(curr_targ, curr_pred):
            if tgt_num == model.END:
                continue
            
            torch_dict = torch.from_numpy(dist_dict[tgt_num, :])
            torch_dict.requires_grad = False

            ew_mult = torch_dict * curr_row
            s_mse += ew_mult.sum()

    s_mse /= (target.shape[0] * target.shape[1])

    return s_mse

def point_type_crossentropy_loss(predicted, target, model):
    cel = nn.NLLLoss()

    n_out_dims = model.n_tokens - model.n_patches + 1

    new_pred = torch.zeros([predicted.shape[0], predicted.shape[1], n_out_dims])
    new_pred[:, :, 1:] = predicted[:, :, -n_out_dims + 1:]
    new_pred[:, :, 0] = 1 - new_pred.sum(axis = 2)
    new_pred = torch.log(new_pred)

    tgt_cpy = torch.clone(target)
    for i, batch in enumerate(target):
        for j, point in enumerate(batch):
            if 0 <= point <= model.n_patches - 1:
                tgt_cpy[i, j] = 0
            else:
                tgt_cpy[i, j] = point - model.n_patches

    out = cel(predicted, tgt_cpy)
    out = torch.exp(out)

    return out


def calculate_loss(predicted, target, model):
    #predicted (batch, n_tokens, n_patches)
    #target (batch, n_patches)

    weights = TRAIN_CONFIG['loss_weights']

    pt_cse = point_type_crossentropy_loss(predicted, target, model)

    s_mse = spatio_mse(predicted, target, model)

    return weights[0] * pt_cse + weights[1] * s_mse