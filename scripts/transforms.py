# Albumentations

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def composed_transforms(dataset_name: str):
    if dataset_name == 'jwst':
        transforms_to_apply = [
        # First, resize the image to be at least the size of the crop
        A.Resize(
            height=256,  # ensure that the image is larger than the crop size
            width=256,
            always_apply=True
        ),
        # Then apply the CenterCrop
        A.CenterCrop(
            height=224,  # after crop resize
            width=224,
            always_apply=True
        ),
        A.ToFloat(max_value=255., always_apply=True),
        ToTensorV2()
        ]
    else: 
        transforms_to_apply = [
            A.Resize(height=224, width=224, always_apply=True),  # resize smaller images
            A.CenterCrop(height=224, width=224, always_apply=True),
            A.ToFloat(max_value=255., always_apply=True),
            ToTensorV2(),
        ]

    composed_transforms = A.Compose(transforms_to_apply)
    return composed_transforms

    
def apply_augmentations(dataset_name: str, dataset):
    augmentation_pipeline = composed_transforms(dataset_name)
    def transforms(examples):

        # np.array to load the PIL image as array for albumentations
        # composed_transforms(image=...) to apply albumentations as per usual
        # list comprehension because examples['image'] is a list with 1 element and should return a list with 1 element (HF convention)
        examples["image"] = [augmentation_pipeline(image=np.array(image.convert("RGB")))['image'] for image in examples["image"]]

        return examples

    return dataset.set_transform(transforms)



