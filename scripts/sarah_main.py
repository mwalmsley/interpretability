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

import loadmodel, transforms, predictions, selection, visualize



# REPO_ID = ""

# Define all dataset str

datasets = ['is_tidal', 
            'decals10',
            'jwst']


tidal_dict = {0: 'not_tidal', 1: 'tidal'}

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
    
    
    dataset_name = 'jwst'


    print("1. Loading model...")
    model = loadmodel.load_model(dataset_name, jwst_loc="C:/Users/ysara/Desktop/repos/1201.ckpt")
    model.eval()

    print("2. Downloading dataset...")
    dataset = load_dataset("mwalmsley/jwst")
    dataset = dataset.with_format('torch')


    print("3. Applying Albumentations transformations...")
    transforms.apply_augmentations(dataset_name, dataset)
    dataloader = torch.utils.data.DataLoader(dataset = dataset['train'], batch_size=16)

    print("4. Making predictions...")
    ce_losses, preds = predictions.make_predictions(model, dataloader)
    sorted_losses, sorted_indices = torch.sort(ce_losses, descending = True)

    print("5. Selecting images based on predictions...")

    # First method: top/bottom n losses
    # n = int(0.05 * dataset['train'].num_rows) # number of samples to extract
    n = 10
    high_ce_indices = selection.select_indices(sorted_indices, n)
    low_ce_indices = selection.select_indices(sorted_indices, n, strategy='bottom')
    
    high_loss_batch = dataset['train'][high_ce_indices]
    low_loss_batch = dataset['train'][low_ce_indices]
    high_loss_preds = preds[high_ce_indices]
    low_loss_preds = preds[low_ce_indices]


    # Second method: model is wrong
    middle_ce_indices = selection.select_indices(sorted_indices, n, strategy = 'middle')
   
    mid_loss_batch = dataset['train'][middle_ce_indices]
    mid_loss_preds = preds[middle_ce_indices]


    print("6. Passing through interpretability methods...")
    batch = high_loss_batch
    preds = high_loss_preds
    # Gradcam
    target_layer = [model.encoder.stages[-1].blocks[-1]]
    gradcam_result = gradcam.apply_gradcam(model, target_layer, batch['image'], method='HiResCam')
    

    # Captum
    ig_viz = IntegratedGradientVisualizer(model=model.to('cpu'))
    attributions = ig_viz.calculate_attributions(images=batch['image'], labels=batch['label'])
    loss_scores = [i[torch.argmax(i)] for i in preds]




    print("7. Visualizing images...")
    labels_dict = globals()[f"{dataset_name}_dict"]

    visualize.visualize_gradcam(batch, gradcam_result, labels_dict, preds)

    visualize.visualize_attr(batch, attributions, labels_dict, loss_scores, preds)


    print(
        'hello world, sarah!'
    )