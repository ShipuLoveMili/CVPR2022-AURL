############################# LIBRARIES ######################################
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

"""=================================================================================================================="""


def get_network(opt):
    """
    Selection function for available networks.
    """
    if 'r3d' in opt.network:
        network = models.video.r3d_18
    elif '2plus1d' in opt.network:
        network = models.video.r2plus1d_18
    elif 'c3d' in opt.network:
        return C3D(fixconvs=opt.fixconvs, nopretrained=opt.nopretrained)
    else:
        raise Exception('Network {} not available!'.format(opt.network))
    return ResNet18(network, fixconvs=opt.fixconvs, nopretrained=opt.nopretrained)


"""=================================================================================================================="""


class ResNet18(nn.Module):
    """
    Container for ResNet50 s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """

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
        #x = x.reshape(bs, nc, -1)
        #x = torch.mean(x, 1)
        x = self.dropout(x)
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
    

def Nceloss(img,word,label,lossfn):
    cosine = F.linear(F.normalize(img, p=2, dim=1), F.normalize(word, p=2, dim=1))
    logits = 10 * cosine
    return lossfn(logits, label)

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
        self.criterion_ce = torch.nn.CrossEntropyLoss(reduction='mean')
        self.criterion_mse = torch.nn.MSELoss(reduction='mean')
    def forward(self, imgbatch, classembedding, label, opt=None, Z_mix=None, la=None, lb=None, lam=None):
        z1 = self.encode(imgbatch)
        z1 = z1.reshape(label.shape[0], z1.shape[0]//label.shape[0], -1)
        z1 = torch.mean(z1, 1)
        z2 = self.encode_words(classembedding)
        if self.training:
           img_w = self.classifier.weight
           txt_w = self.classifier_txts.weight
           if opt.solution==0:
              loss = solution_ours(z1, z2, img_w, txt_w, label, self.criterion_ce)
           return loss, z1, z2
        else:
           return z1, z1, z2


def solution_ours(z1, z2, img_w, txt_w, label, ce):
    Lcon = Nceloss(z1, z2, label, ce)
    Lvideos = Nceloss(z1, img_w, label, ce)
    lam = torch.rand((662,662)).cuda()
    mixed_txts = torch.mm(lam,F.normalize(z2, p=2, dim=1))
    mixed_imgw = torch.mm(lam,F.normalize(img_w, p=2, dim=1))
    global_label = torch.from_numpy(np.array(range(662))).cuda()
    Ltxts = Nceloss(mixed_txts, mixed_imgw, global_label, ce)
    return Lcon + Lvideos + Ltxts


