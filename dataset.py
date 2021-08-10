import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

from saliency import dataset
from config import *

class PathformerData(Dataset):
    #seq, stim, img_emb, seq_patch
    def __init__(self, config, dataset_config, model_config):
        self.data_path = config['data_path']
        self.dataset_name = config['dataset_name']
        self.image_splits = config['image_splits']

        self.n_patches = self.image_splits[0] * self.image_splits[1]

        self.t_seq_length = model_config['t_seq_length']

        self.ds = dataset.SaliencyDataset(config = dataset_config)
        self.ds.load(self.dataset_name)

        self.sequence_directory = os.path.join(self.data_path, "sequence")
        self.stimuli_directory = os.path.join(self.data_path, "stimuli")
        self.image_embedding_directory = os.path.join(self.data_path, "image_embedding")
        self.sequence_patch_directory = os.path.join(self.data_path, "sequence_patch")

        if not os.path.isdir(self.data_path):
            os.mkdir(self.data_path)
        if not os.path.isdir(self.sequence_directory):
            os.mkdir(self.sequence_directory)
        if not os.path.isdir(self.stimuli_directory):
            os.mkdir(self.stimuli_directory)
        if not os.path.isdir(self.image_embedding_directory):
            os.mkdir(self.image_embedding_directory)
        if not os.path.isdir(self.sequence_patch_directory):
            os.mkdir(self.sequence_patch_directory)

    def __len__(self):
        sample_length = len(self.ds.get('stimuli_path'))
        return sample_length

    def __getitem__(self, idx):
        filename = "pathformer_" + str(idx).rjust(5, '0') + ".npy"

        sequence_file_name = os.path.join(self.sequence_directory, filename)
        if os.path.isfile(sequence_file_name):
            sequence = np.load(sequence_file_name, allow_pickle = True)
        else:
            sequence = self.save_sequence(sequence_file_name, idx)
        
        stimuli_file_name = os.path.join(self.stimuli_directory, filename)
        if os.path.isfile(stimuli_file_name):
            stimuli = np.load(stimuli_file_name, allow_pickle = True)
        else:
            stimuli = self.save_stimuli(stimuli_file_name, idx)

        image_embedding_file_name = os.path.join(self.image_embedding_directory, filename)
        if os.path.isfile(image_embedding_file_name):
            image_embedding = np.load(image_embedding_file_name, allow_pickle = True)
        else:
            image_embedding = self.save_image_embedding(image_embedding_file_name, idx)

        sequence_patch_file_name = os.path.join(self.sequence_patch_directory, filename)
        if os.path.isfile(sequence_patch_file_name):
            sequence_patch = np.load(sequence_patch_file_name, allow_pickle = True)
        else:
            sequence_patch = self.save_sequence_patch(sequence_patch_file_name, idx)

        if not torch.is_tensor(stimuli):
            stimuli = torch.from_numpy(stimuli)
        if not torch.is_tensor(image_embedding):
            image_embedding = torch.from_numpy(image_embedding)
        if not torch.is_tensor(sequence_patch):
            sequence_patch = torch.from_numpy(sequence_patch)

        return {"stimuli": stimuli, "image_embedding": image_embedding, "sequence_patch":sequence_patch}

    def save_sequence(self, filename, idx):
        seq = self.ds.get("sequence", percentile = True, modify = 'fix', index = [idx])
        seq = np.squeeze(seq, axis=0)
        np.save(filename, seq)

        return seq

    def save_stimuli(self, filename, idx):
        stim = self.ds.get("stimuli", index = [idx])
        stim = np.squeeze(stim, axis = 0)
        np.save(filename, stim)

        return stim

    def save_image_embedding(self, filename, idx):
        stim = self.ds.get("stimuli", index = [idx])
        stim = np.squeeze(stim, axis = 0)

        height_splits = self.image_splits[0]
        width_splits = self.image_splits[1]

        N = stim.shape[0] // height_splits
        M = stim.shape[1] // width_splits

        tiles = [stim[x:x+M,y:y+N] for x in range(0,stim.shape[0],M) for y in range(0,stim.shape[1],N)]

        #normalizing image
        tiles = np.array(tiles, dtype=np.float32)
        tiles = tiles / 127.5 - 1

        np.save(filename, tiles)

        return tiles

    def save_sequence_patch(self, filename, idx):
        seq = self.ds.get("sequence", percentile = True, modify = 'fix', index = [idx])
        seq = np.squeeze(seq, axis=0)

        height_splits = self.image_splits[0]
        width_splits = self.image_splits[1]

        break_points_height = 1 / height_splits
        break_points_width = 1 / width_splits

        observer_list = list()
        for j, obs in enumerate(seq):
            width_batch = obs[:, 0] // break_points_width
            height_batch = obs[:, 1] // break_points_height
        
            final_batch = height_batch * width_splits + width_batch

            observer_list.append(np.array(final_batch, dtype = np.uint8))
        
        observer_list = np.array(observer_list, dtype = np.object)
        observer_list = self.pad_sequence_patch(observer_list)

        np.save(filename, observer_list, allow_pickle = True)
        
        return observer_list

    def pad_sequence_patch(self, seq):
        self.END = self.n_patches + 2
        self.NONE = self.n_patches + 3
        out = torch.zeros((self.t_seq_length, seq.shape[0]), dtype=torch.long)
        for j, fix in enumerate(seq):
            fix = torch.ByteTensor(fix)
            out[0:len(fix), j] = torch.ByteTensor(fix)
            out[len(fix), j] = self.END
            out[len(fix) + 1:, j] = torch.full([self.t_seq_length - len(fix) - 1], self.NONE, dtype = torch.long)
        #create array thing
        out = out.permute([1, 0])
        return out


class ImageEmbeddings(Dataset):
    def __init__(self, config, fresh_download=False, data_range = "all"):
        #data range: "all" or [0, 200] <- where 0 is start location and 200 is end
        self.data_path = config['data_path']
        self.dataset_name = config['dataset_name']
        self.image_splits = config['image_splits']
        self.data_range = config["data_range"]

        ds = dataset.SaliencyDataset(config=DATASET_CONFIG)
        ds.load(self.dataset_name)

        if data_range == "all":
            self.seq = ds.get('sequence', percentile=True, modify='fix')
            self.stim = ds.get('stimuli', )
            self.stim_paths = ds.get('stimuli_path')
        else:
            rng = data_range
            self.seq = ds.get('sequence', percentile=True, modify='fix', index=range(data_range[0], data_range[1]))
            self.stim = ds.get('stimuli', index = range(data_range[0], data_range[1]))
            self.stim_paths = ds.get('stimuli_path', index = range(data_range[0], data_range[1]))

        if data_range == "all":
            self.image_embedding_path = os.path.join(self.data_path, "image_embeddings.npy")
            self.position_embedding_path = os.path.join(self.data_path, "position_embeddings.npy")
        else:
            self.image_embedding_path = os.path.join(self.data_path, f"image_embeddings_{data_range[0]}.npy")
            self.position_embedding_path = os.path.join(self.data_path, f"position_embeddings_{data_range[0]}.npy")

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
            self.pos_emb = np.load(self.position_embedding_path, allow_pickle=True)
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

        np.save(self.position_embedding_path, pos_emb, allow_pickle = True)
        
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
            #tiles = tiles.reshape(height_splits * width_splits, -1)

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

def get_data_from_batch_number(idx, config, redownload, data_type = "train"):
    if data_type not in ['train', 'test', 'val']:
        raise Exception("Must be train, test, or val")

    batch_size = MODEL_CONFIG['batch_size']

    train_start = IMAGE_EMBEDDING_CONFIG['train_idx'][0]
    test_start = IMAGE_EMBEDDING_CONFIG['test_idx'][0]
    val_start = IMAGE_EMBEDDING_CONFIG['val_idx'][0]

    if data_type == "train":
        if idx >= (IMAGE_EMBEDDING_CONFIG['train_idx'][1] - IMAGE_EMBEDDING_CONFIG['train_idx'][0]) // batch_size:
            raise ValueError('Train IDX not in range')
        else:
            data_range = [train_start + idx * batch_size, train_start + (idx + 1) * batch_size]

    elif data_type == "test":
        if idx >= (IMAGE_EMBEDDING_CONFIG['test_idx'][1] - IMAGE_EMBEDDING_CONFIG['test_idx'][0]) // batch_size:
            raise ValueError('Test IDX not in range')
        else:
            data_range = [test_start + idx * batch_size, test_start + (idx + 1) * batch_size]

    elif data_type == "val":
        if idx >= (IMAGE_EMBEDDING_CONFIG['val_idx'][1] - IMAGE_EMBEDDING_CONFIG['val_idx'][0]) // batch_size:
            raise ValueError('Val IDX not in range')
        else:
            data_range = [val_start + idx * batch_size, val_start + (idx + 1) * batch_size]

    ds = ImageEmbeddings(IMAGE_EMBEDDING_CONFIG, fresh_download = redownload, data_range = data_range)

    return ds.seq, ds.stim, ds.img_emb, ds.pos_emb

def create_target_sequence(seq, model):
    out = torch.clone(seq)
    out = out[:, :, 0:-1]
    begin_pad = torch.full([seq.shape[0], seq.shape[1], 1], model.START)
    out = torch.cat([begin_pad, out], dim = 2)
    '''
    out = torch.zeros((seq.shape[0], seq.shape[1], seq.shape[2]), dtype=torch.long)
    for i, curr in enumerate(seq):
        for j, fix in enumerate(curr):
            fix = torch.ByteTensor(fix)
            out[i, 0:len(fix), j] = torch.ByteTensor(fix)
            out[i, len(fix), j] = model.END
            out[i, len(fix) + 1:, j] = torch.full([model.t_seq_length - len(fix) - 1], model.NONE, dtype = torch.long)
    #create array thing
    '''
    return out