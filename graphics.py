import torch

def display_image_batch(model, stimuli, lowres_stimuli, sequence_patch, display_blur = False):
    model.eval()

    out = model(None, stimuli)

    print(out)