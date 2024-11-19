import os
import sys


# Directory setup, since mine are messed up
zoobot_dir = 'C:/Users/ysara/Desktop/repos/zoobot1'
os.path.isdir(zoobot_dir)
sys.path.insert(0,zoobot_dir)


repo_dir = 'C:/Users/ysara/Desktop/repos/galaxy-datasets'
os.path.isdir(repo_dir)
sys.path.insert(0,repo_dir)

interp_dir = 'C:/Users/ysara/Desktop/repos/new_interpretability_dir/interpretability'
os.path.isdir(interp_dir)
sys.path.insert(0,interp_dir)

gradcam_dir = 'C:/Users/ysara/Desktop/repos/pytorchgradcam'
os.path.isdir(gradcam_dir)
sys.path.insert(0,gradcam_dir)

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
import timm
from torch import nn
from captum.attr import IntegratedGradients

# Locally cloned repos
import zoobot
from zoobot.pytorch.training import finetune
from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule
from galaxy_datasets import transforms as galaxy_datasets_transforms
# from new_interpretability_dir import interpretability # MAY NEED THIS ON A DIFFERENT DEVICE DEPENDING ON DIRECTORY PATH SETUP
import gradcam
from integrated_gradients import IntegratedGradientVisualizer

# refactored scripts
import loadmodel, transforms, predictions, selection, visualize



# REPO_ID = ""

# Define all dataset str

datasets = ['is_tidal', 
            'decals10',
            'jwst']

hf_names = {'is_tidal': 'is-lsb', 'decals10': 'galaxy10_decals', 'jwst': 'jwst'}

is_tidal_dict = {0: 'not_tidal', 1: 'tidal'}

decals10_dict = {0: 'disturbed', 1: 'merging', 2: 'round_smooth', 3: 'in-between_round_smooth', 4: 'cigar-shaped_smooth', 
                 5: 'barred_spiral', 6: 'unbarred_tight_spiral', 7: 'unbarred_loose_spiral', 8: 'edge-on_no_bulge', 9: 'edge-on_with_bulge'}

jwst_dict = {0: 'artifact', 1: 'blob', 2: 'edge-on', 3: 'other-featured', 4: 'protospiral'}


if __name__ == '__main__':


    # Set seed
    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    dataset_name = 'decals10'
    dataset_address = "mwalmsley/" + hf_names[dataset_name]
    labels_dict = globals()[f"{dataset_name}_dict"]


    print("1. Loading model...")
    model = loadmodel.load_model(dataset_name, jwst_loc="C:/Users/ysara/Desktop/repos/1201.ckpt")
    model.eval()

    print("2. Downloading dataset...")
    dataset = load_dataset(dataset_address)
    dataset = dataset.with_format('torch')


    print("3. Applying Albumentations transformations...")
    transforms.apply_augmentations(dataset_name, dataset)
    dataloader = torch.utils.data.DataLoader(dataset = dataset['train'], batch_size=16)

    print("4. Making predictions...")
    ce_losses, preds = predictions.make_predictions(model, dataloader)

    # preds is a tensor of format preds[dataset_size][num_classes]
    sorted_losses, sorted_indices = torch.sort(ce_losses, descending = True)

    print("5. Selecting images based on predictions...")

    # First method: top/bottom n losses
    # n = int(0.05 * dataset['train'].num_rows) # number of samples to extract
    n = 10
    high_ce_indices = selection.select_indices(sorted_indices, n)
    low_ce_indices = selection.select_indices(sorted_indices, n, strategy='bottom')
    
    low_loss_preds = preds[low_ce_indices]
    high_loss_batch = dataset['train'][high_ce_indices]
    low_loss_batch = dataset['train'][low_ce_indices]
    high_loss_preds = preds[high_ce_indices] 

    # Second method: model is wrong
    mismatched_batches = selection.select_mismatches(ce_losses, preds, dataset['train'][:]['label'], labels_dict, dataset)

    print("6. Passing through interpretability methods...")
    for key, name in labels_dict.items():
        batch = mismatched_batches[name]
        preds = batch['preds']
        # Gradcam
        target_layer = [model.encoder.stages[-1].blocks[-1]]
        gradcam_result = gradcam.apply_gradcam(model, target_layer, batch['image'], method='HiResCam')
        

        # Captum
        ig_viz = IntegratedGradientVisualizer(model=model.to('cpu'))
        attributions = ig_viz.calculate_attributions(images=batch['image'], labels=batch['label'])
        loss_scores = [i[torch.argmax(i)] for i in preds]




        save_dir = 'mismatched/'+ dataset_name + '/'+ name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        visualize.visualize_gradcam(batch, gradcam_result, labels_dict, preds, group_size= 1, save_dir = save_dir)

        visualize.visualize_attr(batch, attributions, labels_dict, loss_scores, preds, group_size = 1, save_dir = save_dir)


    print(
        'hello world, sarah!'
    )