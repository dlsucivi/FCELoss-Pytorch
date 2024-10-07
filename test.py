import torch
from torch.autograd import Variable
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score)

def test(model, testloader, class_num, criterion, device):
    """This function tests the model on the test set

    Args:
        model (object): model to be trained
        testloader (object): dataloader for the test set
        class_num (int): number of classes in the dataset
        criterion (object): loss function to be used
        device (str): name of device to use for training

    Returns:
       (float, float, float, float, float, float): returns the test loss, the computed accuracy,
       balanced accuracy, f1 micro, f1 macro, and f1 weighted scores
    """    
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    y_true = torch.LongTensor([]).to(device)
    y_pred = torch.LongTensor([]).to(device)

    for idx, (inputs, targets) in enumerate(testloader):
        with torch.no_grad():
            
            inputs, targets = inputs.to(device), torch.LongTensor(targets).to(device)
            inputs, targets = Variable(inputs), Variable(targets)
            _, output = model(inputs)

            loss = criterion(output, targets)

            test_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

            y_true = torch.cat((y_true, targets.data))
            y_pred = torch.cat((y_pred, predicted))

    y_true = y_true.cpu()
    y_pred = y_pred.cpu()

    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    labels = [x for x in range(class_num)]

    f1_mi = f1_score(y_true, y_pred, labels=labels, average='micro')
    f1_ma = f1_score(y_true, y_pred, labels=labels, average='macro')
    f1_w = f1_score(y_true, y_pred, labels=labels, average='weighted')

    test_loss = test_loss/(len(testloader))
   
    return test_loss, acc, b_acc, f1_mi, f1_ma, f1_w