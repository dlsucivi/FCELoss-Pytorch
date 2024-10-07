import torch.nn as nn
from torchvision import models

def get_backbone(architecture):
    """This model extraccts the backbone from the 
    selected model architecture

    Args:
        architecture (str): model architecture to extract from

    Returns:
        (str[], str[], int): retunrs a list of the layers of the extracted backbone,
        a list of layers in the extracted classifiers, and the number of feature channels. 
    """    
    if architecture == 'ResNet50':
        model = list(models.resnet50(weights='DEFAULT').children()) 
        feature_size = 2048
        backbone = model[:-2]
        classifier = model[-1:]
        
    elif architecture == 'EfficientNet_v2_M':
        model = list(models.efficientnet_v2_m(weights='DEFAULT').children())
        feature_size = 1280
        backbone = model[:-2]
        classifier = model[-1:]

    elif architecture == 'DenseNet169':
        model = list(models.densenet169(weights='DEFAULT').children())
        feature_size = 1664
        backbone = model[:-1]
        classifier = model[-1:]

    return backbone, classifier, feature_size

class backbone_model(nn.Module):
    """This class is the backbone model used. The backbone
    is extracted from an exisiting model architecture to which
    a pooling layer and classifier is added. The forward pass is modified
    such that the extracted features are also returned along with the 
    computed class scores.

    Args:
        architecture (str): model architecture to extract backbone from
        classes_num (int): number of classes in the dataset
    """    
    def __init__(self, architecture, classes_num):

        super(backbone_model, self).__init__()

        backbone, classifier, feature_size = get_backbone(architecture)

        self.backbone = nn.Sequential(*backbone)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(feature_size, classes_num)
       

    def forward(self, x):
        features = self.backbone(x)

        output = self.pool(features)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)

        return features, output