import torch
from fce_loss import fce

def train(model, channels, groups, alpha, trainloader, optimizer, criterion, data_augmentation, device):
    """This function trains the model for one epoch

    Args:
        model (object): model to be trained
        channels (int[]): number of channels assigned to each class group
        groups (int[]): number of consecutive classes with the same number of assigned channels
        alpha (float): weight value to multiply to the FCE loss value
        trainloader (object): dataloader for the train set
        optimizer (object): optimization algorithm for back propagation
        criterion (object): loss function to be used
        data_augmentation (function): applies data augmentation to the data
        device (str): name of device to use for training

    Returns:
        float: returns the training loss of the model for one epoch
    """    
    model.train()
    train_loss = 0

    for idx, (x, y) in enumerate(trainloader):
    
        inputs, targets = data_augmentation(x, torch.LongTensor(y)) 
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        features, output = model(inputs)

        fce_loss = fce(features, targets, channels, groups, criterion, alpha, device)
        ce_loss = criterion(output, targets)

        loss = ce_loss + fce_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss = train_loss/(len(trainloader))
   
    return train_loss