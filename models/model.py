import torchvision
import torch.nn.functional as F 
from torch import nn
from config import config


def loadnet(config):
    if config.model_name=='densenet169':
        return torchvision.models.densenet169(pretrained=True)
    elif config.model_name=='densenet161':
        return torchvision.models.densenet161(pretrained=True)
    elif config.model_name=='densenet121':
        return torchvision.models.densenet121(pretrained=True)
    elif config.model_name=='densenet201':
        return torchvision.models.densenet201(pretrained=True)
    elif config.model_name=='resnet101':
        return torchvision.models.resnet101(pretrained=True)
    elif config.model_name=='resnet152':
        return torchvision.models.resnet152(pretrained=True)
    elif config.model_name=='resnet50':
        return torchvision.models.resnet50(pretrained=True)

def densenet_model():
    class DenseModel(nn.Module):
        def __init__(self, pretrained_model):
            super(DenseModel, self).__init__()
            self.classifier = nn.Linear(pretrained_model.classifier.in_features, config.num_garbages)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()

            self.features = pretrained_model.features
            self.layer1 = pretrained_model.features._modules['denseblock1']
            self.layer2 = pretrained_model.features._modules['denseblock2']
            self.layer3 = pretrained_model.features._modules['denseblock3']
            self.layer4 = pretrained_model.features._modules['denseblock4']

        def forward(self, x):
            features = self.features(x)
            out = F.relu(features, inplace=True)
            out = F.avg_pool2d(out, kernel_size=8).view(features.size(0), -1)
            out = F.relu(out,inplace=True)

            out = F.sigmoid(self.classifier(out))
            
            return out

    return DenseModel(loadnet(config))

def resnet_model():
    class ResModel(nn.Module):
        def __init__(self, pretrained_model):
            super(ResModel, self).__init__()
            self.classifier = nn.Linear(1000, config.num_garbages)
            
            self.resnet_layer = nn.Sequential(*list(pretrained_model.children())[:-2])
            self.transion_layer = nn.ConvTranspose2d(2048, 2048, kernel_size=14, stride=3)
            self.pool_layer = nn.MaxPool2d(32)  
            self.Linear_layer = nn.Linear(2048, 1000)

        def forward(self, x):
            out = self.resnet_layer(x)
            #x = self.transion_layer(x)
            out = self.pool_layer(out)
            #out = F.relu(out, inplace=True)
            out = out.view(out.size(0), -1)
            out = self.Linear_layer(out) 


            out = F.sigmoid(self.classifier(out))
            
            return out

    return ResModel(loadnet(config))





