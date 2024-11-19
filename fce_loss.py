import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random

def assign_remaining_channels(channels_per_class, num_features):
    """This function assigns the remaning channels to the classes with
    with the least number of samples

    Args:
        channels_per_class (numpy.ndarray): the current number of channels assigned to each class
        num_features (int): number of features from the backbone architecture

    Returns:
        numpy.ndarray-int: returns a numpy array containing the 
        number of channels assigned to each class
    """    

    idx = 0
    sorted_indices = channels_per_class.argsort()

    while(channels_per_class.sum() < num_features):
        cur_class_idx = sorted_indices[idx]
        channels_per_class[cur_class_idx] += 1
        idx += 1

        if(idx == len(channels_per_class)):
            idx = 0

    return channels_per_class

def remove_excess_channels(channels_per_class, num_features):
    """This function removes the excess channels from the classes 
    with the most number of samples

    Args:
        channels_per_class (numpy.ndarray): the current number of channels assigned to each class
        num_features (int): number of features from the backbone architecture

    Returns:
        numpy.ndarray-int: returns a numpy array containing 
        the number of channels assigned to each class
    """    
    idx = 1
    sorted_indices = channels_per_class.argsort()

    while(channels_per_class.sum() > num_features):
        cur_class_idx = sorted_indices[-idx]
        cur_channel =  channels_per_class[cur_class_idx]

        if(cur_channel <= 1):
            idx = 1
        else:
            channels_per_class[cur_class_idx] -= 1
            idx += 1

    return channels_per_class

def compute_class_groups(channels_per_class):
    """This function groups consecutive classes with the same
    number of channels assigned to them

    Args:
        channels_per_class (numpy.ndarray): number of channels assigned to each class

    Returns:
        (int[], int[]): returns a tuple of integer lists contaning the number of channels
        and the formed groups of classes.
    """    
    groups = []
    channels = []
    cur_group_count = 1
    num_classes = len(channels_per_class)

    for i in range(1,num_classes):
        if(channels_per_class[i] != channels_per_class[i-1]):
            groups.append(cur_group_count)
            channels.append(channels_per_class[i-1])
            cur_group_count = 1
        else:
            cur_group_count += 1

    groups.append(cur_group_count)
    channels.append(channels_per_class[i])

    return channels, groups

def compute_inverse_proporotions(samples_per_class):
    """This function computes for the inverse proportion of
    the number of samples for each class

    Args:
        samples_per_class (numpy.ndarray): number of samples for each class

    Returns:
        float[]: returns the computed inverse proportions for each class
    """    

    inverse_proportions = 1 / samples_per_class

    return inverse_proportions


def compute_class_equity(samples_per_class, num_features):
    """This function assigns channels to each class inversely
    proportional to the number of samples they contain

    Args:
        samples_per_class (numpy.ndarray): number of samples for each class
        num_features (int): number of features from the backbone architecture_

    Returns:
        (int[], int[]): returns a tuple of integer lists contaning the number of channels
        and the number of consecutive classes assigned with the same number of channels.
    """    
    total_sample_count = samples_per_class.sum()
    class_proportions = samples_per_class / total_sample_count
    channels_per_class = np.floor(class_proportions * num_features).astype(int)
    channels_per_class[channels_per_class == 0] = 1

    total_channel_count = channels_per_class.sum()

    if(total_channel_count > num_features):
        channels_per_class = remove_excess_channels(channels_per_class, num_features)

    elif(total_channel_count < num_features):
        channels_per_class = assign_remaining_channels(channels_per_class, num_features)

    channels, groups = compute_class_groups(channels_per_class)

    return channels, groups

def get_mask(batch_size, channels, groups, device):
    """This function creates a channel mask for each group of classes

    Args:
        batch_size (int): number of samples in a batch_
        channels (int[]): number of channels assigned to each class group
        groups (int[]): number of consecutive classes with the same number of assigned channels
        device (str): name of device to use for training

    Returns:
        int[][]: returns a 2D list containing integer values 0 or 1
        representing the mask
    """    
    total_channels = np.sum(np.array(channels) * np.array(groups))
    mask = []

    for i in range(len(channels)):
        if(channels[i] < 3):
            foo = [1] * channels[i]
        else:
            difference = channels[i] - 2
            foo = [1] * 2 + [0] * difference
            
        for j in range(groups[i]):
            random.shuffle(foo)
            mask += foo

    mask = [mask for i in range(batch_size)]
    mask = np.array(mask).astype("float32")
    mask = mask.reshape(batch_size,total_channels,1,1)
    mask = torch.from_numpy(mask)
    mask = mask.to(device)
    mask = Variable(mask)

    return mask

def fce(x, targets, channels, groups, criterion, alpha, device):
    """This function computes for the fair channel enhancement loss

    Args:
        x (tensor): input with dimensions NxCxHxW
        targets (tensor): ground-truth labels for input
        channels (int[]): number of channels assigned to each class group
        groups (int[]): number of consecutive classes with the same number of assigned channels
        criterion (object): loss function to be used as part of the FCE loss
        alpha (float): weight value to multiply to the FCE loss value
        device (str): name of device to use for training

    Returns:
        float: returns the computed FCE loss
    """

    n, c, h, w = x.size()
    channel_idxs = [0]
    temp = np.array(groups) * np.array(channels)
    for i in range(len(groups)):
        channel_idxs.append(sum(temp[:i + 1]))

    mask = get_mask(n, channels, groups, device)

    features = mask * x  
    features_group = []
    for i in range(1, len(channel_idxs)):
        features_group.append(features[:, channel_idxs[i - 1]:channel_idxs[i]])

    fce_output = []
    for i in range(len(channels)):
        features = features_group[i]
        features = F.max_pool2d(features.view(n, -1, h * w), 
                                kernel_size=(channels[i], 1), 
                                stride=(channels[i], 1))
        fce_output.append(features)

    fce_output = torch.cat(fce_output, dim=1).view(n, -1, h, w)
    fce_output = nn.AdaptiveAvgPool2d((1, 1))(fce_output).view(n, -1)  

    fce_loss = criterion(fce_output, targets) * alpha

    return fce_loss