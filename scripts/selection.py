# Selection

import torch
import random
import numpy as np

def select_indices(sorted_indices, n, strategy='top', percent=0):
    """
    Selects n indices from the dataset based on the specified strategy.

    Args:
        dataset (Dataset): The dataset to sample from.
        n (int): The number of indices to select.
        strategy (str): The strategy to use. Options are 'top', 'bottom', 'middle', 'percent', and 'random'.
        percent (float): Percentage for 'percent' strategy (value between 0 and 1).
        
    Returns:
        List[int]: Selected sub-list of sorted_indices.
    """
    total_length = len(sorted_indices)
    all_indices = torch.arange(total_length)

    if strategy == 'top':
        selected_indices = all_indices[:n]
    
    elif strategy == 'bottom':
        selected_indices = all_indices[-n:]
    
    elif strategy == 'middle':
        start = (total_length - n) // 2
        selected_indices = all_indices[start:start + n]
    
    elif strategy == 'percent':
        assert 0 < percent <= 1, "Percent must be between 0 and 1 for 'percent' strategy."
        n = int(total_length * percent)
        selected_indices = all_indices[:n]
    
    elif strategy == 'random':
        selected_indices = torch.tensor(random.sample(range(total_length), n))
    
    else:
        raise ValueError("Invalid strategy. Choose from 'top', 'bottom', 'middle', 'percent', or 'random'.")
    
    indices = sorted_indices[selected_indices.tolist()]
    
    return indices



def select_mismatches(ce_losses, preds, labels, label_dict, dataset):
    """
    Sorts cross-entropy losses by class label and identifies misclassified indices.

    """
    batches = {}

    # loop through each unique label from dataset['labels']
    for class_label in np.unique(labels):

        label_indices = np.where(labels == class_label)[0]
        preds_for_class = preds[label_indices].cpu() # softmax probs for each selected image.
        
        
        predicted_classes = [torch.argmax(i) for i in preds_for_class]

        misclassified = np.where(predicted_classes != class_label)[0] # returns the imgs in the class where model has misclassified
        misclassified = label_indices[misclassified] # returns the index in the dataset of the misclassified img
       
    #    creating a batch of mismatches per class
        if label_dict and class_label in label_dict:
                    category_name = label_dict[class_label]
                    batches[category_name] = {
                        'id': misclassified, 
                        'image': dataset['train'][misclassified]['image'], 
                        'preds': preds[misclassified],
                        'loss': ce_losses[misclassified],
                        'label': dataset['train'][misclassified]['label']
                    }

    return batches
