# Visualize images from a batch

import matplotlib.pyplot as plt
from matplotlib import ticker
import torch
import seaborn as sns

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


def visualize_attr(input_batch, attributions, labels_dict, scores, preds):
    ncols = 2
    nrows = len(attributions)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, nrows * 3))

    for idx in range(len(attributions)):
        # Left column: Display raw image
        img_ax = axes[idx, 0]
        img = input_batch['image'][idx].cpu().numpy().transpose(1, 2, 0)
        img_ax.imshow(img)
        pred_label = labels_dict[torch.argmax(preds[idx]).item()]
        true_label = labels_dict[input_batch['label'][idx]]
        img_ax.set_title(f"Pred: {pred_label}     Label: {true_label}")
        img_ax.axis('off')
        
        # Right column: Display attribution map
        attri_ax = axes[idx, 1]
        attri_plot = attributions[idx][0, 0, :, :]
        attri_ax.imshow(attri_plot, cmap='RdBu', vmin=-0.25, vmax=0.25)
        attri_ax.set_title(f"Score: {'{:.2f}'.format(scores[idx])}")
        attri_ax.axis('off')
    
    plt.subplots_adjust(hspace=0.5)
    plt.show()


def visualize_gradcam(input_batch, gradcam, labels_dict, preds):
    positive_cmap = sns.color_palette("mako", as_cmap=True)
    negative_cmap = sns.color_palette("rocket", as_cmap=True)
    fig, rows = plt.subplots(ncols=4, nrows=len(input_batch['image']), figsize=(6, 14))

        # pred_ind = torch.argmax(preds[im_n]) # model prediction category
        # ax.text(30, -10, '{:.2f}'.format(pred_ind))
        # ax.text(200, -10, '{:.0f}'.format(input_batch['label'][im_n].item()))

    for im_n in range(len(input_batch['image'])):
        ax0, ax1, ax2, ax3 = rows[im_n]
        im = input_batch['image'][im_n]

        if input_batch['label'][im_n] ==torch.argmax(preds[im_n]):
            cmap = positive_cmap
        else:
            cmap = negative_cmap

        im_viz = im.numpy().transpose(1, 2, 0)
        output_viz = gradcam[im_n].squeeze()
        # output_viz = np.where(output_viz<0.15, np.nan, output_viz)  # skip blacks
        # clip_upper = np.percentile(output_viz[output_viz > 0], 0.999)
        # clip_upper = 0.3
        # output_viz = np.clip(output_viz, 0, clip_upper)

        # Show the images on each subplot
        ax0.imshow(im_viz)
        ax0.axis('off')

        ax1.imshow(output_viz, cmap=cmap, vmin=0., vmax=1.)
        ax1.axis('off')

        ax2.imshow(im_viz)
        ax2.imshow(output_viz, alpha=.6, cmap=cmap, vmin=0., vmax=1.)
        ax2.axis('off')

        ax3.hist(output_viz.flatten(), range=(0, 1), bins=30)
        ax3.yaxis.set_major_locator(ticker.NullLocator())
        ax3.xaxis.set_major_locator(ticker.NullLocator())

        ax0.text(30, -10, 'vol={}, p={:.2f}, pred={}'.format(
        labels_dict[input_batch['label'][im_n]],
        preds.cpu().detach().numpy()[im_n][input_batch['label'][im_n]],
        labels_dict[torch.argmax(preds[im_n]).item()])
        )