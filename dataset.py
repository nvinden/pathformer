import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

from saliency import dataset
from config import *

class ImageEmbeddings(Dataset):
    def __init__(self, config, fresh_download=False):
        self.data_path = config['data_path']
        self.dataset_name = config['dataset_name']
        self.image_splits = config['image_splits']
        self.data_range = config["data_range"]

        ds = dataset.SaliencyDataset(config=DATASET_CONFIG)
        ds.load(self.dataset_name)

        if self.data_range == "all":
            self.seq = ds.get('sequence', percentile=True, modify='fix')
            self.stim = ds.get('stimuli', )
            self.stim_paths = ds.get('stimuli_path')
        else:
            rng = self.data_range
            self.seq = ds.get('sequence', percentile=True, modify='fix', index = range(rng))
            self.stim = ds.get('stimuli', index = range(rng))
            self.stim_paths = ds.get('stimuli_path', index = range(rng))

        self.image_embedding_path = os.path.join(self.data_path, "image_embeddings.npy")
        self.position_embedding_path = os.path.join(self.data_path, "position_embeddings.npy")

        if fresh_download or not os.path.isfile(self.image_embedding_path):
            img_emb = self._create_img_embeddings()
            self.img_emb = img_emb
        elif os.path.isfile(self.image_embedding_path):
            self.img_emb = np.load(self.image_embedding_path)
            if self.img_emb.shape[0] != self.seq.shape[0]:
                self.img_emb = self._create_img_embeddings()
        else:
            raise FileNotFoundError

        if fresh_download or not os.path.isfile(self.position_embedding_path):
            pos_emb = self._create_pos_embeddings()
            self.pos_emb = pos_emb
        elif os.path.isfile(self.position_embedding_path):
            self.pos_emb = np.load(self.position_embedding_path)
            if self.pos_emb.shape[0] != self.seq.shape[0]:
                self.pos_emb = self._create_pos_embeddings()
        else:
            raise FileNotFoundError

    def _create_pos_embeddings(self):
        print("Creating a fresh position embedding directory...")

        height_splits = self.image_splits[0]
        width_splits = self.image_splits[1]

        break_points_height = 1 / height_splits
        break_points_width = 1 / width_splits

        pos_emb = list()
        for i, seq in enumerate(self.seq):
            observer_list = list()
            for j, obs in enumerate(seq):
                width_batch = obs[:, 0] // break_points_width
                height_batch = obs[:, 1] // break_points_height
            
                final_batch = height_batch * width_splits + width_batch

                observer_list.append(np.array(final_batch, dtype = np.uint8))

            pos_emb.append(observer_list)
        
        return np.array(pos_emb)

            
    def _create_img_embeddings(self):
        print("Creating a fresh image embedding directory...")

        height_splits = self.image_splits[0]
        width_splits = self.image_splits[1]
        
        img_emb = list()
        for i, img in enumerate(self.stim):
            N = img.shape[0] // height_splits
            M = img.shape[1] // width_splits

            tiles = [img[x:x+M,y:y+N] for x in range(0,img.shape[0],M) for y in range(0,img.shape[1],N)]

            #normalizing image
            tiles = np.array(tiles, dtype=np.float32)
            tiles = tiles / 127.5 - 1
            tiles = tiles.reshape(height_splits * width_splits, -1)

            img_emb.append(tiles)
        img_emb = np.array(img_emb)
        np.save(self.image_embedding_path, img_emb)
        return img_emb

    def __len__(self):
        return self.seq.shape[0]

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.seq[idx])
        stim = torch.from_numpy(self.stim[idx])
        img_emb = torch.from_numpy(self.img_emb[idx])
        seq_patch = torch.from_numpy(self.pos_emb[idx])
        
        return {"seq": seq, "seq_patch": seq_patch, "stim": stim, "img_emb": img_emb}

def create_batches(ds, batch_size):
    batches = list()
    for i in range(0, len(ds), batch_size):
        current = ds[i:i + batch_size]
        batches.append(current)

    return np.array(batches)

def create_target_sequence(seq, model):
    out = torch.zeros((seq.shape[0], model.t_seq_length, seq.shape[1]), dtype=torch.long)
    for i, curr in enumerate(seq):
        for j, fix in enumerate(curr):
            fix = torch.ByteTensor(fix)
            out[i, 0:len(fix), j] = torch.ByteTensor(fix)
            out[i, len(fix), j] = model.END
            out[i, len(fix) + 1:, j] = torch.full([model.t_seq_length - len(fix) - 1], model.NONE, dtype = torch.long)
    #create array thing
    return out