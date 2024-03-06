import torch

def reduce_brightness_center_circle(image_batch, diameter, reduction_factor):

    center_h, center_w = image_batch.shape[2] // 2, image_batch.shape[3] // 2
    radius = diameter / 2


    mask = torch.ones_like(image_batch)


    for i in range(image_batch.shape[2]):
        for j in range(image_batch.shape[3]):
            if (i - center_h) ** 2 + (j - center_w) ** 2 <= radius ** 2:
                mask[:, :, i, j] *= reduction_factor

    return image_batch * mask