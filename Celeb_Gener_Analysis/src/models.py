import torch.nn as nn
import pretrainedmodels
import torch.nn.functional as F

class Resnet34(nn.Module):

    def __init__(self, pretrained='imagenet'):
        super(Resnet34, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained=None)


        self.l0 = nn.Linear(512, 1)
        self.act1 = nn.Sigmoid()
        

    def forward(self, x):

        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        
        l0 = self.l0(x)
        l0 = self.act1(l0)

        return l0

class Resnet50(nn.Module):

    def __init__(self, pretrained='imagenet'):
        super(Resnet50, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained=None)


        self.l0 = nn.Linear(2048, 1)
        self.act1 = nn.Sigmoid()
        

    def forward(self, x):

        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        
        l0 = self.l0(x)
        l0 = self.act1(l0)

        return l0

class Resnet101(nn.Module):

    def __init__(self, pretrained='imagenet'):
        super(Resnet101, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet101'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet101'](pretrained=None)


        self.l0 = nn.Linear(2048, 1)
        self.act1 = nn.Sigmoid()
        

    def forward(self, x):

        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        
        l0 = self.l0(x)
        l0 = self.act1(l0)

        return l0