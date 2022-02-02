import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TokenToPosition(nn.Module):
    def __init__(self, batch_size = 32, dropout = 0.1):
        super(TokenToPosition, self).__init__()

        self.dropout = dropout
        self.batch_size = batch_size

        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride=1, padding=1)
        self.linear = nn.Linear(256 * 6 * 6, 2, dtype = torch.float)

    def forward(self, image):
        if image.shape[3] == 3:
            image.transpose_(1, 3)

        out = self.conv1(image)
        out = self.pool(F.relu(out))
        out = self.conv2(out)
        out = self.pool(F.relu(out))
        out = self.conv3(out)
        out = self.pool(F.relu(out))
        out = self.conv4(out)
        out = self.pool(F.relu(out))
        out = out.view(self.batch_size, -1)
        out = self.linear(out)

        out = torch.clamp(out, min = 0.0, max = 1.0)

        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PathFormer(nn.Module):
    def __init__(self, config, img_config):
        super(PathFormer, self).__init__()

        self.dataset_name = img_config['dataset_name']

        if self.dataset_name == "CAT2000":
            weights = [1376, 1531, 2796, 4476, 4353, 3570, 1638, 440, 483, 3248,
                9386, 20239, 20653, 10718, 4205, 770, 506, 4489, 15888, 60240, 65158,
                18121, 6386, 972, 479, 4129, 13937, 43417, 61881, 15929, 6292, 802, 
                292, 2297, 7788, 15524, 17259, 8572, 3459, 519, 133, 961, 2875, 9023, 
                6122, 3069, 1605, 393, 1, 0, 28501, 195598]
            self.ds_token_weights = torch.tensor(weights, device = device) / sum(weights)

        #image parameters
        self.image_splits = img_config['image_splits']
        self.n_patches = self.image_splits[0] * self.image_splits[1]
        self.height = img_config['height']
        self.width = img_config['width']
        self.patch_height = self.height // self.image_splits[0]
        self.patch_width = self.width // self.image_splits[1]
        self.blur = img_config['blur']

        #embedding parameters
        self.batch_size = config["batch_size"]
        self.D = config["D"]
        self.img_patch_area = config["img_patch_area"]
        self.image_embedding_dimension = config['image_embedding_dimension']
        self.decoder_embedding_dimension = config['decoder_embedding_dimension']

        #transformer parameters
        self.t_seq_length = config["t_seq_length"]
        self.t_n_head = config["t_n_head"]
        self.t_n_encoder_layers = config["t_n_encoder_layers"]
        self.t_n_decoder_layers = config["t_n_decoder_layers"]
        self.t_dim_feedforward = config["t_dim_feedforward"]
        self.t_dropout = config["t_dropout"]
        self.t_activation = config["t_activation"]
        self.t_batch_first = config["t_batch_first"]

        self.n_tokens = self.n_patches + 4

        self.dec_mask = self.generate_square_subsequent_mask(sz = self.t_seq_length)

        #macros for positional information
        self.NO_FIX = self.n_patches
        self.START = self.n_patches + 1
        self.END = self.n_patches + 2
        self.NONE = self.n_patches + 3

        #convolutional linear embedding
        self.vgg19 = config["vgg19"]
        if self.vgg19:
            torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
            self.vgg19_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True)
            self.encoder_linear = nn.Linear(1000, self.D)
        else:
            self.pool = nn.MaxPool2d(2, 2)
            self.encoder_conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 7, stride=1, padding=3)
            self.encoder_conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride=1, padding=2)
            self.encoder_conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 5, stride=1, padding=2)
            self.encoder_conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride=1, padding=1)
            self.encoder_linear = nn.Linear(256 * 6 * 6, self.D)

        self.dropout = nn.Dropout(p = self.t_dropout)

        self.encoder_positional_embedding = nn.Embedding(self.n_tokens, self.decoder_embedding_dimension)
        self.decoder_positional_embedding = nn.Embedding(self.n_tokens, self.decoder_embedding_dimension)

        self.encoder_positional_encoding = PositionalEncoding(self.D, self.t_dropout, max_len = self.batch_size)
        self.decoder_positional_encoding = PositionalEncoding(self.decoder_embedding_dimension, self.t_dropout, max_len = self.batch_size)

        self.transformer = nn.Transformer(d_model = self.D,
                                            nhead = self.t_n_head,
                                            num_encoder_layers = self.t_n_encoder_layers,
                                            num_decoder_layers = self.t_n_decoder_layers,
                                            dim_feedforward = self.t_dim_feedforward,
                                            dropout = self.t_dropout,
                                            activation = self.t_activation,
                                            batch_first = self.t_batch_first)

        self.output_linear = nn.Linear(self.D, self.n_tokens)

    def create_embeddable_seq_encoder(self, seq):
        out = dict()
        for i, fix in enumerate(seq):
            if fix in out.keys():
                out[fix].append(i)
            else:
                app = list()
                app.append(i)
                out[fix] = app

        return out

    def create_dec_embeddings(self, seq_dec):
        dec_embeddings = torch.zeros((self.batch_size, self.t_seq_length, self.decoder_embedding_dimension), dtype = torch.float32, requires_grad  = False, device = device)
        for i in range(self.batch_size):
            idx = 0
            curr = seq_dec[i]
            dec_embeddings[i, idx] = self.decoder_positional_embedding(torch.LongTensor([self.START], device = device)).squeeze(0)
            idx += 1
            for j in range(1, len(curr)):
                fix_patch = curr[j]
                dec_embeddings[i, idx] = self.decoder_positional_embedding(torch.LongTensor([fix_patch], device = device)).squeeze(0)
                idx += 1
        return dec_embeddings 

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device = device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, seq, img_patches):
        if not torch.is_tensor(img_patches):
            img_patches = torch.from_numpy(img_patches, device = device)
        img_patches.requires_grad = False

        #convolution
        img_patches = img_patches.permute([0, 1, 4, 2, 3])
        img_patches = img_patches.view([-1, 3, self.patch_height, self.patch_width])

        if self.vgg19:
            enc_emb = self.vgg19_model(img_patches)
            enc_emb = self.encoder_linear(enc_emb)
        else:
            enc_emb = self.encoder_conv1(img_patches)
            enc_emb = self.pool(F.relu(enc_emb))
            enc_emb = self.encoder_conv2(enc_emb)
            enc_emb = self.pool(F.relu(enc_emb))
            enc_emb = self.encoder_conv3(enc_emb)
            enc_emb = self.pool(F.relu(enc_emb))
            enc_emb = self.encoder_conv4(enc_emb)
            enc_emb = self.pool(F.relu(enc_emb))
            enc_emb = enc_emb.view(self.batch_size * self.n_patches, -1)
            enc_emb = self.encoder_linear(self.dropout(enc_emb))
            enc_emb = enc_emb.view(self.batch_size, self.n_patches, -1)

        enc_emb = self.encoder_positional_encoding(enc_emb)

        if self.training == True:
            #decoder
            #(32, 25, 768)
            dec_emb = self.create_dec_embeddings(seq)
            dec_emb = self.decoder_positional_encoding(dec_emb)
            dec_pad = torch.zeros(self.batch_size, self.t_seq_length, self.D - dec_emb.shape[2], device = device)
            dec_emb = torch.cat([dec_emb, dec_pad], axis = 2)

            tgt_padding_mask = (seq == self.NONE)

            out = self.transformer(enc_emb, dec_emb, tgt_mask = self.dec_mask, tgt_key_padding_mask = tgt_padding_mask)
            out = self.output_linear(out)
            out = F.softmax(out, dim = -1)

            out_tokens = torch.argmax(out, dim = -1)

            return out, out_tokens

        #evaluation
        elif self.training == False:
            dec_tokens = torch.full([self.batch_size, self.t_seq_length], self.NONE, device = device)
            dec_tokens[:, 0] = self.START

            dec_tokens_out = torch.full([self.batch_size, self.t_seq_length], self.NONE, device = device)

            for gaze_no in range(self.t_seq_length):
                dec_emb = self.create_dec_embeddings(dec_tokens)
                dec_emb = self.decoder_positional_encoding(dec_emb)
                dec_pad = torch.zeros(self.batch_size, self.t_seq_length, self.D - dec_emb.shape[2], device = device)
                dec_emb = torch.cat([dec_emb, dec_pad], axis = 2)

                out = self.transformer(enc_emb, dec_emb, tgt_mask = self.dec_mask)
                out = self.output_linear(out)
                out = F.softmax(out, dim = -1)

                out_tokens = torch.argmax(out, dim = -1)

                dec_tokens_out[:, gaze_no] = out_tokens[:, gaze_no]
                if not gaze_no >= self.t_seq_length - 1:
                    dec_tokens[:, gaze_no + 1] = out_tokens[:, gaze_no]
            
            return dec_tokens_out
