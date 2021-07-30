import torch
import torch.nn as nn
import torch.nn.functional as F

import math

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
    def __init__(self, config, img_config, train_method):
        super(PathFormer, self).__init__()

        self.train_method = train_method

        assert self.train_method in ["on_self", "on_pic"]

        #image parameters
        self.image_splits = img_config['image_splits']
        self.n_patches = self.image_splits[0] * self.image_splits[1]
        self.height = img_config['height']
        self.width = img_config['width']
        self.patch_height = self.height // self.image_splits[0]
        self.patch_width = self.width // self.image_splits[1]

        #embedding parameters
        self.batch_size = config["batch_size"]
        self.D = config["D"]
        self.img_patch_area = config["img_patch_area"]
        self.image_embedding_dimension = config['image_embedding_dimension']
        self.position_embedding_dimension = config['position_embedding_dimension']

        assert self.image_embedding_dimension + self.position_embedding_dimension == self.D
        #transformer parameters
        #self.t_d_model = config["t_d_model"]
        self.t_n_head = config["t_n_head"]
        self.t_n_encoder_layers = config["t_n_encoder_layers"]
        self.t_n_decoder_layers = config["t_n_decoder_layers"]
        self.t_dim_feedfordward = config["t_dim_feedfordward"]
        self.t_dropout = config["t_dropout"]
        self.t_activation = config["t_activation"]
        self.t_batch_first = config["t_batch_first"]

        self.n_tokens = self.n_patches + 4

        self.dec_mask = self.generate_square_subsequent_mask(sz = self.n_patches)

        #macros for positional information
        self.NO_FIX = self.n_patches
        self.START = self.n_patches + 1
        self.END = self.n_patches + 2
        self.NONE = self.n_patches + 3

        self.encoder_linear_embedding = nn.Linear(self.img_patch_area, self.D)
        #self.encoder_positional_embedding = nn.Embedding(self.n_tokens, self.position_embedding_dimension)
        self.decoder_positional_embedding = nn.Embedding(self.n_tokens, self.D)

        self.positional_encoding = PositionalEncoding(self.D, self.t_dropout, max_len = self.n_patches)

        self.transformer = nn.Transformer(d_model = self.D,
                                            nhead = self.t_n_head,
                                            num_encoder_layers = self.t_n_encoder_layers,
                                            num_decoder_layers = self.t_n_decoder_layers,
                                            dim_feedforward = self.t_dim_feedfordward,
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

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, seq_enc, seq_dec, img_patches):
        img_patches = torch.from_numpy(img_patches)
        img_patches.requires_grad = False

        enc_embeddings = self.encoder_linear_embedding(img_patches)

        #FOR EMBEDDING
        #token: n_patches (48) = no fixations
        #       n_patches + 1 (49) = <START>
        #       n_patches + 2 (50) = <END>
        #       n_patches + 3 (51) = <EMPTY>

        #decoder
        dec_embeddings = torch.zeros((self.batch_size, self.n_patches, self.D), dtype = torch.float32, requires_grad  = False)
        for i in range(self.batch_size):
            idx = 0
            curr = seq_dec[i]
            dec_embeddings[i, idx] = self.decoder_positional_embedding(torch.LongTensor([self.START])).squeeze(0)
            idx += 1
            for fix_patch in curr:
                dec_embeddings[i, idx] = self.decoder_positional_embedding(torch.LongTensor([fix_patch])).squeeze(0)
                idx += 1
            dec_embeddings[i, idx] = self.decoder_positional_embedding(torch.LongTensor([self.END])).squeeze(0)
            idx += 1
            for end_patch in range(idx, self.n_patches):
                dec_embeddings[i, end_patch] = self.decoder_positional_embedding(torch.LongTensor([self.NONE])).squeeze(0)

        enc_embeddings = self.positional_encoding(enc_embeddings)
        dec_embeddings = self.positional_encoding(dec_embeddings)
        
        out = self.transformer(enc_embeddings, dec_embeddings, tgt_mask = self.dec_mask)
        out = self.output_linear(out)
        out = F.softmax(out, dim = -1)

        return out