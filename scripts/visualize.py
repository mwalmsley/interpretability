# Visualize images from a batch

import matplotlib.pyplot as plt
from matplotlib import ticker
import torch
import seaborn as sns
import math
import os

def visualize_batch(input_batch, labels_dict, preds):
    num_images = len(input_batch['image'])
    fig, axes = plt.subplots(ncols=num_images, figsize=(20, 4))

    for im_n in range(num_images):
        ax = axes[im_n]
        ax.imshow(input_batch['image'][im_n].cpu().numpy().transpose(1, 2, 0))
        pred_ind = torch.argmax(preds[im_n]) # model prediction category
        ax.text(30, -10, '{:.2f}'.format(pred_ind))
        ax.text(200, -10, '{}'.format(labels_dict[im_n].item()))
        ax.axis('off')
    plt.show()


def visualize_gradcam(input_batch, gradcam, labels_dict, preds, group_size=4, save_dir = False):
    positive_cmap = sns.color_palette("mako", as_cmap=True)
    negative_cmap = sns.color_palette("rocket", as_cmap=True)

    batch_size = len(input_batch['image'])
    num_chunks = math.ceil(batch_size / group_size)

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * group_size
        end_idx = min(start_idx + group_size, batch_size)
        chunk_size = end_idx - start_idx

        ncols = 4  # Each image has 4 subplots
        nrows = chunk_size

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows))

        # If chunk_size is 1, make axes a 2D list for consistent indexing
        if chunk_size == 1:
            axes = [axes]

        for i, idx in enumerate(range(start_idx, end_idx)):
            row_axes = axes[i] if chunk_size > 1 else axes[0]
            ax0, ax1, ax2, ax3 = row_axes

            im = input_batch['image'][idx]
            cmap = positive_cmap if input_batch['label'][idx] == torch.argmax(preds[idx]) else negative_cmap

            im_viz = im.numpy().transpose(1, 2, 0)  # Convert from CHW to HWC
            output_viz = gradcam[idx].squeeze()

            # Show the images on each subplot
            ax0.imshow(im_viz)
            ax0.axis('off')

            ax1.imshow(output_viz, cmap=cmap, vmin=0., vmax=1.)
            ax1.axis('off')

            ax2.imshow(im_viz)
            ax2.imshow(output_viz, alpha=0.6, cmap=cmap, vmin=0., vmax=1.)
            ax2.axis('off')

            ax3.hist(output_viz.flatten(), range=(0, 1), bins=30)
            ax3.yaxis.set_major_locator(ticker.NullLocator())
            ax3.xaxis.set_major_locator(ticker.NullLocator())

            # Add annotations
            ax0.text(30, -10, 'ID = {}, vol={}, p={:.2f}, pred={}'.format(
                input_batch['id'][idx],
                labels_dict[input_batch['label'][idx]],
                preds.cpu().detach().numpy()[idx][input_batch['label'][idx]],
                labels_dict[torch.argmax(preds[idx]).item()]
            ))

            if save_dir:
                fig_path = os.path.join(save_dir, f"gradcam_{input_batch['id'][idx]}.png")
                fig.savefig(fig_path)
                plt.close(fig)

        plt.subplots_adjust(hspace=0.5)




def visualize_attr(input_batch, attributions, labels_dict, scores, preds, group_size=4, save_dir = False):
    batch_size = len(attributions)
    num_chunks = math.ceil(batch_size / group_size)  # number of groups to plot

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * group_size
        end_idx = min(start_idx + group_size, batch_size)
        chunk_size = end_idx - start_idx

        ncols = 2
        nrows = chunk_size  

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, nrows * 3))

        # If chunk_size is 1, make axes a 2D list for consistent indexing
        if chunk_size == 1:
            axes = [axes]

        for i, idx in enumerate(range(start_idx, end_idx)):
            img_ax, attri_ax = axes[i] if chunk_size > 1 else axes[0]

            # Left column: Display raw image
            img = input_batch['image'][idx].cpu().numpy().transpose(1, 2, 0)
            pred_label = labels_dict[torch.argmax(preds[idx]).item()]
            true_label = labels_dict[input_batch['label'][idx]]
            id = input_batch['id'][idx]
            img_ax.imshow(img)
            img_ax.set_title(f"ID: {id}      Pred: {pred_label}     Label: {true_label}")
            img_ax.axis('off')

            # Right column: Display attribution map
            attri_plot = attributions[idx][0, 0, :, :]
            attri_ax.imshow(attri_plot, cmap='RdBu', vmin=-0.25, vmax=0.25)
            attri_ax.set_title(f"Score: {'{:.2f}'.format(scores[idx])}")
            attri_ax.axis('off')

            if save_dir:
                fig_path = os.path.join(save_dir, f"captum_{id}.png")
                fig.savefig(fig_path)
                plt.close(fig)

        plt.subplots_adjust(hspace=0.5)

            