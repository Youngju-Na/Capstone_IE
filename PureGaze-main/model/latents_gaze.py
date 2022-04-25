import torch
import torch.nn as nn
import numpy as np
import math
import modules
import torch.utils.model_zoo as model_zoo
import dct
from torchvision import models
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        ## create dct models
        feature_extractor = models.resnet101(pretrained=True)

        self.feature_extractor = nn.Sequential(
            feature_extractor.conv1,
            feature_extractor.bn1,
            feature_extractor.relu,
            feature_extractor.maxpool,
            feature_extractor.layer1,
            feature_extractor.layer2,
            feature_extractor.layer3,
            feature_extractor.layer4,
            feature_extractor.avgpool
        )
        
        # self.reduction = nn.Linear(2048, 300)

        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        
        self.output_layer = nn.Sequential(
            nn.Linear(1024+2048, 512),
            # nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2)
        )


        
    def forward(self, x_in,  trained=True):
        ## face
        feature = self.feature_extractor(x_in['face'])
        face_feature = feature.view(feature.shape[0], -1)
        ## latent
        latents = x_in["latents"]
        latents =  latents.view(latents.shape[0], -1)
        latents = latents[:, 1536+512:2560+512]

        feature = torch.cat((face_feature, latents), 1)
        gaze = self.output_layer(feature)
        return gaze

class Gelossop():
    def __init__(self, w1=1, w2=1):

        self.gloss = torch.nn.MSELoss().cuda()
        self.w1 = w1
        self.w2 = w2
        

    def __call__(self, pre_gaze, GT_gaze):
        geloss = self.gloss(pre_gaze, GT_gaze)
        return self.w1 * geloss


