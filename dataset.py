import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

import torchvision
from torchvision import transforms

from PIL import Image

from saliency import dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PathformerTokenData(Dataset):
    def __init__(self, IMAGE_EMBEDDING_CONFIG, DATASET_CONFIG, MODEL_CONFIG, create_data = False):
        self.data_path = os.path.join(DATASET_CONFIG['data_path'], "PathFormerTokens")
        self.dataset_name = IMAGE_EMBEDDING_CONFIG['dataset_name']
        self.image_splits = IMAGE_EMBEDDING_CONFIG['image_splits']

        self.n_patches = self.image_splits[0] * self.image_splits[1]
        self.n_images = IMAGE_EMBEDDING_CONFIG["n_images"]

        self.t_seq_length = MODEL_CONFIG['t_seq_length']

        self.ds = PathformerData(IMAGE_EMBEDDING_CONFIG, DATASET_CONFIG, MODEL_CONFIG)

        self.seq_pos_directory = os.path.join(self.data_path, "seq_pos")
        self.img_directory = os.path.join(self.data_path, "img")

        self.LENGTH = 98700

        if create_data:
            if not os.path.isdir(self.data_path):
                os.mkdir(self.data_path)
            if not os.path.isdir(self.seq_pos_directory):
                os.mkdir(self.seq_pos_directory)
            if not os.path.isdir(self.img_directory):
                os.mkdir(self.img_directory)

            self.create_data()

    def __len__(self):
        return self.LENGTH

    def __getitem__(self, idx):
        seq_pos_save_path = os.path.join(self.seq_pos_directory, f"seq_pos" + str(idx) + ".npy")
        img_save_path = os.path.join(self.img_directory, f"img" + str(idx) + ".npy")

        seq_pos = np.load(seq_pos_save_path)
        img = np.load(img_save_path)

        if not torch.is_tensor(seq_pos):
            seq_pos = torch.from_numpy(seq_pos)
        if not torch.is_tensor(img):
            img = torch.from_numpy(img)

        return {"seq_pos": seq_pos, "img": img}

    def create_data(self):
        loader = DataLoader(self.ds, batch_size = self.n_images, shuffle = True, drop_last = True)

        for data in loader:
            sequence = data['sequence']
            stim = data['stimuli']
            img_patches = data['image_embedding']
            seq_patch = data['sequence_patch']

            n_patches = img_patches.shape[1]
            img_splits = IMAGE_EMBEDDING_CONFIG['image_splits']

            #creating full tape:
            tape = list()
            for image_seq_patch, image_seq, image_stim in zip(seq_patch, sequence, img_patches):
                for obs_seq_patch, obs_seq in zip(image_seq_patch, image_seq):
                    for point_seq_patch, point_seq in zip(obs_seq_patch, obs_seq):
                        if point_seq_patch >= n_patches:
                            continue
                        pos = point_seq[:2]
                        
                        x_patch_pos = point_seq_patch % img_splits[1]
                        y_patch_pos = point_seq_patch // img_splits[1]

                        x_min_range = 1 / img_splits[1] * x_patch_pos
                        x_max_range = 1 / img_splits[1] * (x_patch_pos + 1)

                        y_min_range = 1 / img_splits[0] * y_patch_pos
                        y_max_range = 1 / img_splits[0] * (y_patch_pos + 1)

                        pos[0] = (pos[0] - x_min_range) / (x_max_range - x_min_range)
                        pos[1] = (pos[1] - y_min_range) / (y_max_range - y_min_range)

                        data_add = {"seq_pos": pos, "img": image_stim[point_seq_patch]}
                        tape.append(data_add)

            for i, data in enumerate(tape):
                seq_pos = data['seq_pos']
                img = data['img']

                seq_pos_save_path = os.path.join(self.seq_pos_directory, f"seq_pos" + str(i) + ".npy")
                img_save_path = os.path.join(self.img_directory, f"img" + str(i) + ".npy")

                seq_pos = seq_pos.float()

                np.save(seq_pos_save_path, seq_pos)
                np.save(img_save_path, img)

class PathformerData(Dataset):
    #seq, stim, img_emb, seq_patch
    def __init__(self, config, dataset_config, model_config):
        self.data_path = config['data_path'] + "_" + config["dataset_name"]

        self.dataset_name = config['dataset_name']
        self.image_splits = config['image_splits']

        if self.dataset_name in ['CAT2000', 'OSIE', 'MIT1003']:
            self.fix_amount = 15
        else:
            assert Exception

        self.n_patches = self.image_splits[0] * self.image_splits[1]

        self.height = config['height']
        self.width = config['width']

        self.t_seq_length = model_config['t_seq_length']

        self.ds = dataset.SaliencyDataset(config = dataset_config)
        self.ds.load(self.dataset_name)   

        self.sequence_directory = os.path.join(self.data_path, "sequence")
        self.stimuli_directory = os.path.join(self.data_path, "stimuli")
        self.lowres_stimuli_directory = os.path.join(self.data_path, "lowres_stimuli")
        self.image_embedding_directory = os.path.join(self.data_path, "image_embedding")
        self.sequence_patch_directory = os.path.join(self.data_path, "sequence_patch")

        if not os.path.isdir(self.data_path):
            os.mkdir(self.data_path)
        if not os.path.isdir(self.sequence_directory):
            os.mkdir(self.sequence_directory)
        if not os.path.isdir(self.lowres_stimuli_directory):
            os.mkdir(self.lowres_stimuli_directory)
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

        lowres_stimuli_file_name = os.path.join(self.lowres_stimuli_directory, filename)
        if os.path.isfile(lowres_stimuli_file_name):
            lowres_stimuli = np.load(lowres_stimuli_file_name, allow_pickle = True)
        else:
            lowres_stimuli = self.save_lowres_stimuli(lowres_stimuli_file_name, idx)

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

        if not torch.is_tensor(sequence):
            sequence = torch.from_numpy(sequence)
        if not torch.is_tensor(stimuli):
            stimuli = torch.from_numpy(stimuli)
        if not torch.is_tensor(image_embedding):
            image_embedding = torch.from_numpy(image_embedding)
        if not torch.is_tensor(sequence_patch):
            sequence_patch = torch.from_numpy(sequence_patch)
        if not torch.is_tensor(lowres_stimuli):
            lowres_stimuli = torch.from_numpy(lowres_stimuli)

        sequence = sequence.to(device)

        return {"sequence": sequence, "stimuli": stimuli, "image_embedding": image_embedding, "sequence_patch":sequence_patch, "lowres_stimuli": lowres_stimuli}

    def save_lowres_stimuli(self, filename, idx):
        RESCALE_FACTOR = 0.1

        stim = self.save_stimuli(filename, idx, save = False)
        stim = stim.permute(2, 0, 1)

        resize = transforms.Compose([
            transforms.Resize(size = (int(stim.shape[1] * RESCALE_FACTOR), int(stim.shape[2] * RESCALE_FACTOR))),
            transforms.Resize(size = (int(stim.shape[1]), int(stim.shape[2])))
        ])

        stim = resize(stim)
        stim = stim.permute(1, 2, 0)

        height_splits = self.image_splits[0]
        width_splits = self.image_splits[1]

        N = stim.shape[0] // height_splits
        M = stim.shape[1] // width_splits

        stim = stim.numpy()

        tiles = [stim[x:x+M,y:y+N] for x in range(0,stim.shape[0],M) for y in range(0,stim.shape[1],N)]

        #normalizing image
        tiles = np.array(tiles, dtype=np.float32)
        tiles = tiles / 127.5 - 1

        np.save(filename, tiles)

        return tiles

    def save_sequence(self, filename, idx, save = True):
        seq = self.ds.get("sequence", percentile = True, modify = 'fix', index = range(idx, idx + 1), start = idx)
        seq = np.squeeze(seq, axis=0)

        if self.dataset_name in ["CAT2000", "MIT1003"]:
            if seq.shape[0] > self.fix_amount:
                seq = seq[:self.fix_amount]
            elif seq.shape[0] < self.fix_amount:
                while seq.shape[0] < self.fix_amount:
                    next_seq = seq[-1].astype('O').copy()
                    seq_list = seq.tolist()
                    seq_list.append(next_seq)
                    seq = np.array(seq_list, dtype = np.object)
            assert len(seq) == self.fix_amount

        seq_columns = 5

        out_tensor = np.empty([seq.shape[0], self.t_seq_length, seq_columns], dtype = float)
        pad_tensor = torch.full((self.t_seq_length, seq_columns), float('nan'))

        for i in range(len(seq)):
            path = seq[i]
            if len(path) > self.t_seq_length:
                path = path[:self.t_seq_length]
            empty_spaces = self.t_seq_length - path.shape[0]
            out_tensor[i] = np.concatenate((path, pad_tensor[:empty_spaces]), axis = 0)

        if save == True:
            np.save(filename, out_tensor)

        return out_tensor

    def save_stimuli(self, filename, idx, save = True):
        stim = self.ds.get("stimuli", index = range(idx, idx + 1), start = idx)
        stim = np.squeeze(stim, axis = 0)

        if self.dataset_name in ["CAT2000", "MIT1003"]:
            '''
            stim = torch.from_numpy(stim)
            stim = stim.permute(2, 0, 1)
            resize = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size = (self.height, self.width)),
            ])
            stim = resize(stim)
            stim = stim.permute(1, 2, 0)
            '''
            im = Image.fromarray(np.uint8(stim))
            im.save("Test.jpg")
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size = (self.height, self.width)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            stim = preprocess(im)
            stim = stim.permute(1, 2, 0)

        if save == True:
            np.save(filename, stim)

        return stim

    def save_image_embedding(self, filename, idx):
        stim = self.save_stimuli(filename, idx, save = False)

        height_splits = self.image_splits[0]
        width_splits = self.image_splits[1]

        N = stim.shape[0] // height_splits
        M = stim.shape[1] // width_splits

        stim = stim.numpy()

        tiles = [stim[x:x+M,y:y+N] for x in range(0,stim.shape[0],M) for y in range(0,stim.shape[1],N)]

        #normalizing image
        tiles = np.array(tiles, dtype=np.float32)

        np.save(filename, tiles)

        return tiles

    def save_sequence_patch(self, filename, idx):
        seq = self.ds.get("sequence", percentile = True, modify = 'fix', index = range(idx, idx + 1), start = idx)
        seq = np.squeeze(seq, axis=0)

        if self.dataset_name in ["CAT2000", "MIT1003"]:
            if seq.shape[0] > self.fix_amount:
                seq = seq[:15]
            elif seq.shape[0] < self.fix_amount:
                while seq.shape[0] < self.fix_amount:
                    next_seq = seq[-1].astype('O').copy()
                    seq_list = seq.tolist()
                    seq_list.append(next_seq)
                    seq = np.array(seq_list, np.object)
            assert len(seq) == self.fix_amount

        height_splits = self.image_splits[0]
        width_splits = self.image_splits[1]

        break_points_height = 1 / height_splits
        break_points_width = 1 / width_splits

        observer_list = list()
        for j, obs in enumerate(seq):
            if obs.shape[0] >= self.t_seq_length - 1:
                obs = obs[:self.t_seq_length - 1]

            width_batch = obs[:, 0] // break_points_width
            height_batch = obs[:, 1] // break_points_height
        
            final_batch = height_batch * width_splits + width_batch
            final_batch = np.clip(final_batch, a_min = 0, a_max = self.n_patches + 3)

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
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            for k in range(seq.shape[2]):
                if out[i, j , k] == model.NONE:
                    out[i, j , k] = model.END
                elif k == out.shape[2] - 1:
                    out[i, j , k] = model.END
    return out