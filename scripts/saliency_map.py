import numpy as np
import torch
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, LayerCAM, FullGrad
from pytorch_grad_cam.utils import model_targets

def saliency_vis(method, model, target_layer, input_batch, input_labels, preds):
    cam = method(model=model, target_layers=target_layer)
    output = [cam(input_tensor=torch.from_numpy(np.expand_dims(im, 0)), targets=None) for im in input_batch]

    fig, rows = plt.subplots(ncols=2, nrows=len(input_batch), figsize=(6, 14))

    for im_n in range(len(input_batch)):
        ax0, ax1 = rows[im_n]
        im = input_batch[im_n]

        # Show the images on each subplot
        ax0.imshow(im.numpy().transpose(1, 2, 0)/255.)
        ax0.axis('off')
        ax1.imshow(output[im_n].squeeze())
        ax1.axis('off')

        ax0.text(30, -10, 'vol={:.0f}, p={:.2f}'.format(
            input_labels[im_n],
            preds.detach().numpy()[im_n][1])
        )