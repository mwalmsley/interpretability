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