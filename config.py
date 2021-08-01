DATASET_CONFIG = {
	'data_path' : "Datasets",
	'dataset_json': 'saliency/data/dataset.json',
	'auto_download' : True
}

IMAGE_EMBEDDING_CONFIG = {
	'data_path': "Datasets/image_embeddings",
	'dataset_name': "OSIE",
	'height': 600,
	'width': 800,
	'image_splits': [6, 8],
	'data_range': 32,
}

MODEL_CONFIG = {
	'batch_size': 32,
	"D": 768,
	"img_patch_area": 30000,

	"image_embedding_dimension": 748,
	"position_embedding_dimension": 20,

	"t_seq_length": 25,
	"t_n_head": 6,
	"t_n_encoder_layers": 1,
	"t_n_decoder_layers": 1,
	"t_dim_feedforward": 100,
	"t_dropout": 0.0,
	"t_activation": "relu",
	"t_batch_first": True

}

TRAIN_CONFIG = {
	"n_epochs": 100,
	"lr": 0.5,
	"loss_weights": [1, 1]
}