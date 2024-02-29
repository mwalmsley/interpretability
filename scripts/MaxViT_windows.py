import numpy as np
from matplotlib import pyplot as plt
import torch
import scipy.sparse as sp
import pandas as pd
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import timm
from timm.models.layers import PatchEmbed
from pprint import pprint
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# CustomDataset processes the images
class CustomDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths.iloc[idx])
        if self.transform:
            image = self.transform(image)
        return image
    
def process_img_maxvit(train_catalog, file_locations, num_img):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])

    file_paths = train_catalog[file_locations].iloc[:num_img]

    image_dataset = CustomDataset(file_paths=file_paths, transform=transform)

    dataloader = DataLoader(image_dataset, batch_size=1, num_workers=1)

    image_tensors = []

    for batch in dataloader:
        image_tensors.append(batch)

    # 'image_tensors' will contain a list of tensors, each with a size of (1, 3, 224, 224)
    # You can stack them to create a single tensor of size (n, 3, 224, 224)
    image_tensor = torch.cat(image_tensors, dim=0)
    
    return image_tensor

def maxvit_layer_names(model):
    train_nodes, eval_nodes = get_graph_node_names(model, tracer_kwargs={'leaf_modules': [PatchEmbed]})
    pprint(eval_nodes)

class MaxvitVisualizer:
    """Defines the variables and functions to visualize the grid and block attention mechanisms in MaxViT.
    Can specify which layer in MaxViT to look at."""
    
    def __init__(self, grid_layer, block_layer):
        self.grid_layer = grid_layer
        self.block_layer = block_layer


    def grid_attn(model, grid_layer, single_image_tensor):
        feature_extractor_grid = create_feature_extractor(
            model, return_nodes=[grid_layer],
            tracer_kwargs={'leaf_modules': [PatchEmbed]})
        with torch.no_grad():
            return feature_extractor_grid(255*single_image_tensor)


    def block_attn(model, block_layer, single_image_tensor):
        feature_extractor_block = create_feature_extractor(
            model, return_nodes=[block_layer],
            tracer_kwargs={'leaf_modules': [PatchEmbed]})
        with torch.no_grad():
            return feature_extractor_block(255*single_image_tensor)
    
    def visualizer(grid_attn, block_attn, grid_layer, block_layer):
        slice_to_visualize = grid_attn[grid_layer].sum(axis=(0, -1)).cpu().numpy()
        grid_to_visualize = block_attn[block_layer].sum(axis=(0, -1)).cpu().numpy()


        plt.imshow(slice_to_visualize, cmap='viridis')
        plt.title('Block Attention, First Layer')
        cbar1 = plt.colorbar()
        plt.show()
        im = plt.imshow(grid_to_visualize, cmap='viridis')
        plt.title('Grid Attention, First Layer')


        cbar2 = plt.colorbar()
        im.set_clim(vmin=slice_to_visualize.min(), vmax=slice_to_visualize.max())
        # cbar1.set_clim(vmin=slice_to_visualize.min(), vmax=slice_to_visualize.max())
        plt.show()

        
