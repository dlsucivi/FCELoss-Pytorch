import torch
import os
import numpy as np

def mkdir(directory):
    """This function makes a directory with the given name if one
    is not created yet

    Args:
        directory (str): name of the directory
    """    
    if not os.path.exists(directory):
        os.makedirs(directory)


def write_print(path, text):
    """Displays text in console and saves in text file

    Arguments:
        path {string} -- path to text file
        text {string} -- text to display and save
    """
    file = open(path, 'a')
    file.write(text + '\n')
    file.close()
    print(text)

def collate(batch):
    """This function is used to create the bacthes of input

    Args:
        batch (object): the batch of images with their corresponding classes

    Returns:
        object: returns the processed batch
    """    
    images = []
    targets = []
    for sample in batch:
        images.append(sample[0])
        targets.append(int(sample[1]))
    return torch.stack(images, 0), targets

def count_class_samples(dataset, num_classes):
    """This function counts the number of samples for each class

    Args:
        dataset (object): the dataset used
        num_classes (int): how many classes are in the dataset

    Returns:
        numpy.ndarray: returns a numpy array containing the number of samples
        for each class
    """    
    labels = np.zeros(num_classes, dtype=int)
    
    for _, target in dataset:
        labels[int(target)] += 1
    
    return labels

