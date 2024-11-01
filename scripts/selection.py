# Selection

import torch
import random

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


