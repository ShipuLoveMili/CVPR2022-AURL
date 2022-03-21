import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

def get_network():
    network = models.video.r2plus1d_18
    return ResNet18(network, fixconvs=False, nopretrained=False)



class ResNet18(nn.Module):

    def __init__(self, network, fixconvs=False, nopretrained=True):
        super(ResNet18, self).__init__()
        self.model = network(pretrained=nopretrained)
        if fixconvs:
            for param in self.model.parameters():
                param.requires_grad = False

        self.regressor = nn.Linear(self.model.fc.in_features, 300)
        self.dropout = torch.nn.Dropout(p=0.05)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))

    def forward(self, x):
        bs, nc, ch, l, h, w = x.shape
        x = x.reshape(bs*nc, ch, l, h, w)
        x = self.model(x)
        x = x.view(bs*nc, -1)
        return x
    def get_output_dim(self):
        return 512

class Backbone_words(nn.Module):
    def __init__(self):
        super(Backbone_words, self).__init__()
        self.regressor = nn.Linear(300, 512)
    def forward(self,embedding):
        embedding = self.regressor(embedding)
        return embedding
    def get_output_dim(self):
        return 512
    
class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(ProjectionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        self.l3 = nn.Sequential(
            nn.Linear(mid_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        return x
    
class CosineLayer(torch.nn.Module):

    def __init__(self, in_features=2048, out_features=128):
        super(CosineLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.parameter.Parameter(torch.cuda.FloatTensor(out_features, in_features))

        torch.nn.init.uniform_(self.weight)

    def forward(self, inputs):
        cosine = F.linear(F.normalize(inputs, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        cosine = 10 * cosine
        return cosine



class AURL(nn.Module):
    def __init__(self, imgbackbone, wordsbackbone):
        super(AURL, self).__init__()
        
        self.backbone = imgbackbone
        self.backbone_words = wordsbackbone
        backbone_outdims = self.backbone.get_output_dim()
        backbone_words_outdims = self.backbone_words.get_output_dim()
        self.projection = ProjectionMLP(backbone_outdims,2048,2048)
        self.projection_words = ProjectionMLP(backbone_words_outdims,2048,2048)
        self.encode = nn.Sequential(
            self.backbone,
            self.projection
        )
        self.encode_words = nn.Sequential(
            self.backbone_words,
            self.projection_words
        )
        self.classifier = CosineLayer(2048,662)
        self.classifier_txts = CosineLayer(2048,662)
    def forward(self, imgbatch):#, classembedding):
        z1 = self.encode(imgbatch)
        z1 = z1.reshape(imgbatch.shape[0], z1.shape[0]//imgbatch.shape[0], -1)
        z1 = torch.mean(z1, 1)
        #z2 = self.encode_words(classembedding)
        return z1#, z2




