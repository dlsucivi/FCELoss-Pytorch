import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import os

from dataset import SD
from train import train
from test import test
from augmentations import Augmentations, BaseTransform
from model import backbone_model
from utils import mkdir, write_print, collate, count_class_samples
from datetime import datetime
from fce_loss import compute_class_equity, compute_inverse_proporotions


#########################
#   DATASET PARAMETERS  #
#########################
DATASET = 'SD-198'
DATA_PATH = '../../Datasets/SD/SD/'
VERSION = 'SD-198'
SPLIT = '8-2'
FOLD = '1'
IMG_SIZE = 224
mean = (104, 117, 123)

#########################
#   MODEL PARAMETERS    #
#########################
CLASS_NUM = 198 #200
BACKBONE = 'DenseNet169'
ALPHA = 1.5
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
OPTIMIZER = 'SGD'

#########################
#  TRAINING PARAMETERS  #
#########################
LR = 0.001
EPOCHS = 120
BATCH_SIZE = 32
pretrained = True
isTraining = True
weights_folder = ''
weights_file = ''
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

#########################
#     PREPARE DATASET   #
#########################
print('==> Preparing data..')
trainset = SD(data_path=DATA_PATH,
              ver=VERSION,
              split=SPLIT,
              fold=FOLD,
              mode='train',
              new_size=IMG_SIZE,
              image_transform=(Augmentations(IMG_SIZE, mean)))
trainloader = DataLoader(dataset=trainset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=8,
                        drop_last=True,
                        collate_fn=collate)

testset = SD(data_path=DATA_PATH,
             ver=VERSION,
             split=SPLIT,
             fold=FOLD,
             mode='test',
             new_size=IMG_SIZE,
             image_transform=(BaseTransform(IMG_SIZE, mean)))

testloader = DataLoader(dataset=testset, 
                        batch_size=BATCH_SIZE, 
                        shuffle=True, 
                        num_workers=8, 
                        collate_fn=collate)

#########################
#   DATA AUGMENTATION   #
#########################
print('==> Augmennting Data..')
def get_labels(batch):
    return batch[1]

cutmix = v2.CutMix(num_classes=CLASS_NUM, labels_getter=get_labels)
data_augmentation = v2.RandomChoice([cutmix])

#########################
#      FCE PARAMETERS   #
#########################
print('==> Preparing Equity..')
count_per_class = count_class_samples(trainset, CLASS_NUM)
inverse_proportions = compute_inverse_proporotions(count_per_class)


#########################
#    INITIALIZE MODEL   #
#########################
if(BACKBONE == 'ResNet50'):
   CHANNEL_NUM, GROUP_NUM = compute_class_equity(inverse_proportions, 2048)
if(BACKBONE == 'EfficientNet_v2_M'):
    CHANNEL_NUM, GROUP_NUM = compute_class_equity(inverse_proportions, 1280)
if(BACKBONE == 'DenseNet169'):
    CHANNEL_NUM, GROUP_NUM = compute_class_equity(inverse_proportions, 1664)

model = backbone_model(BACKBONE, CLASS_NUM)
model.backbone.to(DEVICE)
model.pool.to(DEVICE)
model.classifier.to(DEVICE)

#########################
#      LOAD MODEL       #
#########################
if(not isTraining):
    saved_state_dict = torch.load(os.path.join('weights', weights_folder, weights_file))
    model.load_state_dict(saved_state_dict)

optimize = optim.SGD([
                        {'params': model.backbone.parameters(),   'lr': LR},
                        {'params': model.classifier.parameters(), 'lr': LR},
                     ],
                      momentum=0.9, weight_decay=5e-4)

if __name__ == '__main__':
    max_val_acc = 0
    max_b_acc = 0
    max_f1_mi = 0
    max_f1_ma = 0
    max_f1_w = 0
    best_epoch = 0

    if(isTraining):
        weights_path = os.path.join('weights', str(datetime.now()).replace(':', '_'))
        mkdir(weights_path)

        log = f"------------ Options -------------\n"\
        f'dataset: {DATASET}\n'\
        f'train set path: {trainset.list_path}\n'\
        f'test set path: {testset.list_path}\n'\
        f'version: {VERSION}\n'\
        f'split: {SPLIT}\n'\
        f'fold: {FOLD}\n'\
        f'no. of classes; {CLASS_NUM}\n'\
        f'model: {BACKBONE}\n'\
        f'optimizer: {OPTIMIZER}\n'\
        f'image size: {IMG_SIZE}\n'\
        f'epochs: {EPOCHS}\n'\
        f'batch_size: {BATCH_SIZE}\n'\
        f'learning rate: {LR}\n'\
        f'pretrained: {pretrained}\n'\
        f'channels: {CHANNEL_NUM}\n'\
        f'groups: {GROUP_NUM}\n'\
        f'-------------- End ----------------\n\n'\
        f"------------ Model Architecture -------------\n"\
        f'{model}\n'
        f'-------------- End ----------------\n\n'\
        
        write_print(os.path.join(weights_path, 'record.txt'), log)

        for epoch in range(1, EPOCHS+1):

            if epoch == 100 or epoch == 110:
                optimize.param_groups[0]['lr'] /= 10
                optimize.param_groups[1]['lr'] /= 10

            train_loss = train(model, CHANNEL_NUM, GROUP_NUM, ALPHA, 
                               trainloader, optimize, criterion, data_augmentation, 
                               DEVICE)
            
            test_loss, test_acc, b_acc, f1_mi, f1_ma, f1_w = test(model ,testloader, CLASS_NUM, 
                                                                  criterion, DEVICE)

            if test_acc > max_val_acc:
                max_val_acc = test_acc
                max_b_acc = b_acc
                max_f1_mi = f1_mi
                max_f1_ma = f1_ma
                max_f1_w = f1_w
                best_epoch = epoch
                path = os.path.join(weights_path, f'best.pth')
                torch.save(model.state_dict(), path)

            if(epoch % 10 == 0):
                path = os.path.join(weights_path,f'{epoch}.pth')
                torch.save(model.state_dict(), path)

            log = "Epoch [{}/{}]: train_loss {:.4f}, acc {:.4f}, b_acc {:.4f}, f1_mi {:.4f}, f1_ma {:.4f}, f1_w {:.4f}".format(epoch,
                EPOCHS,
                train_loss,
                test_acc,
                b_acc,
                f1_mi,
                f1_ma,
                f1_w)

            write_print(os.path.join(weights_path, 'record.txt'), log)

            print(f"Best epoch: {best_epoch}, max_val_acc {max_val_acc:.4f}, b_acc {max_b_acc:.4f}, f1_mi {max_f1_mi:.4f}, f1_ma {max_f1_ma:.4f}, f1_w {max_f1_w:.4f}\n")
    else:
        test_loss, test_acc, b_acc, f1_mi, f1_ma, f1_w = test(model, testloader, CLASS_NUM, 
                                                              criterion, DEVICE)
        
        print(f"acc {test_acc:.4f}, b_acc {b_acc:.4f}, f1_mi {f1_mi:.4f}, f1_ma {f1_ma:.4f}, f1_w {f1_w:.4f}")


