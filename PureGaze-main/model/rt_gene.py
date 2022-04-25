import torch
import torch.nn as nn
import numpy as np
import math
import modules
import torch.utils.model_zoo as model_zoo
import dct
from torchvision import models
import torchvision
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        ## create dct models
        vgg16ForLeft = torchvision.models.vgg16(pretrained=True)
        vgg16ForRight = torchvision.models.vgg16(pretrained=True)

        self.leftEyeNet = vgg16ForLeft.features
        self.rightEyeNet = vgg16ForRight.features

        for param in self.leftEyeNet.parameters():
            param.requires_grad = True
        for param in self.rightEyeNet.parameters():
            param.requires_grad = True

        self.leftPool = nn.AdaptiveAvgPool2d(1)
        self.rightPool = nn.AdaptiveAvgPool2d(1)

        self.leftFC = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        self.rightFC = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )

        self.totalFC1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )

        self.totalFC2 = nn.Sequential(
            nn.Linear(514, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

        
    def forward(self, x_in,  trained=True):
        face = x_in["face"]
        headlabel = x_in["headlabel"]
        left_eye = x_in["left_eye"]
        right_eye = x_in["right_eye"]

        leftFeature = self.leftEyeNet(left_eye)
        rightFeature = self.rightEyeNet(right_eye)

        leftFeature = self.leftPool(leftFeature)
        rightFeature = self.rightPool(rightFeature)

        leftFeature = leftFeature.view(leftFeature.size(0), -1)
        rightFeature = rightFeature.view(rightFeature.size(0), -1)

        leftFeature = self.leftFC(leftFeature)
        rightFeature = self.rightFC(rightFeature)

        feature = torch.cat((leftFeature, rightFeature), 1)

        feature = self.totalFC1(feature)
        feature = torch.cat((feature,  headlabel), 1)

        gaze = self.totalFC2(feature)

        return gaze

class Gelossop():
    def __init__(self, w1=1, w2=1):

        self.gloss = torch.nn.L1Loss().cuda()
        self.recloss = torch.nn.MSELoss().cuda()
        self.w1 = w1
        self.w2 = w2
        

    def __call__(self, pre_gaze, GT_gaze):
        geloss = self.gloss(pre_gaze, GT_gaze)

        return self.w1 * geloss


