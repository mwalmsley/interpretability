import torch
from captum.attr import IntegratedGradients
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


def visualize_mismatched_images(mismatched_images, mismatched_scores, mismatched_labels):
    ncols = 8
    nrows = (len(mismatched_images) + ncols - 1) // ncols

    for row in range(nrows):
        fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(20, 4))
        axes = axes.flatten() if nrows > 1 else [axes]

        for col in range(ncols):
            idx = row * ncols + col
            if idx < len(mismatched_images):
                ax = axes[col]
                ax.imshow(mismatched_images[idx].numpy().transpose(1, 2, 0) / 255.)
                score_text = '{:.2f}'.format(mismatched_scores[idx])
                ax.text(10, 20, f'Score: {score_text}\nLabel: {mismatched_labels[idx]}', fontsize=12, color='white', backgroundcolor='black')
                ax.axis('off')
            else:
                axes[col].axis('off')
        plt.show()

class IntegratedGradientVisualizer:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.ig = IntegratedGradients(model)
        self.default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                              [(0, '#ffffff'),
                                                               (0.25, '#000000'),
                                                               (1, '#000000')], N=256)

    def calculate_attributions(self, images, labels, use_positive_attributions_only=False):
        all_attributions = []
        for i, img_tensor in enumerate(images):
            input_sample = img_tensor.unsqueeze(0)
            baseline = torch.zeros_like(input_sample)
            target_class = int(labels[i])

            attributions, delta = self.ig.attribute(input_sample, baseline, target=target_class,
                                                    return_convergence_delta=True)
            print(f'Convergence Delta for image {i}:', delta)

            attributions_numpy = attributions.numpy()
            if use_positive_attributions_only:
                attributions_numpy[attributions_numpy < 0] = 0
            all_attributions.append(attributions_numpy)

        return all_attributions

    def visualize_attributions(self, attributions, labels, scores):
        ncols = 8
        nrows = (len(attributions) + ncols - 1) // ncols

        for row in range(nrows):
            fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(20, 4))
            for col, ax in enumerate(axes):
                idx = row * ncols + col
                if idx < len(attributions):
                    attri_plot = attributions[idx][0, 0, :, :]
                    ax.imshow(attri_plot, cmap='RdBu', vmin=-0.25, vmax=0.25)
                    ax.set_title(f"Score: {'{:.2f}'.format(scores[idx])}\nLabel: {labels[idx]}")
                    ax.axis('off')
                else:
                    ax.axis('off')
            plt.show()

    def run(self, images, labels, scores, use_positive_attributions_only=False):
        attributions = self.calculate_attributions(images, labels, use_positive_attributions_only)
        self.visualize_attributions(attributions, labels, scores)
