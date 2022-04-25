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
        self.dct = dct.DCT_2D()
        self.idct = dct.IDCT_2D()
        ## create resnet models
        _left_model = models.resnet18(pretrained=True)
        _right_model = models.resnet18(pretrained=True)
        _left_dct_model = models.resnet18(pretrained=True)
        _right_dct_model = models.resnet18(pretrained=True)
        _face_dct_model = models.resnet18(pretrained=True)

        fc_in_num = _left_model.fc.in_features

        # remove the last ConvBRelu layer
        self.left_features = nn.Sequential(
            _left_model.conv1,
            _left_model.bn1,
            _left_model.relu,
            _left_model.maxpool,
            _left_model.layer1,
            _left_model.layer2,
            _left_model.layer3,
            _left_model.layer4,
            _left_model.avgpool
        )

        self.right_features = nn.Sequential(
            _right_model.conv1,
            _right_model.bn1,
            _right_model.relu,
            _right_model.maxpool,
            _right_model.layer1,
            _right_model.layer2,
            _right_model.layer3,
            _right_model.layer4,
            _right_model.avgpool
        )
        self.left_dct_features = nn.Sequential(
            _left_dct_model.conv1,
            _left_dct_model.bn1,
            _left_dct_model.relu,
            _left_dct_model.maxpool,
            _left_dct_model.layer1,
            _left_dct_model.layer2,
            _left_dct_model.layer3,
            _left_dct_model.layer4,
            _left_dct_model.avgpool
        )
        self.right_dct_features = nn.Sequential(
            _right_dct_model.conv1,
            _right_dct_model.bn1,
            _right_dct_model.relu,
            _right_dct_model.maxpool,
            _right_dct_model.layer1,
            _right_dct_model.layer2,
            _right_dct_model.layer3,
            _right_dct_model.layer4,
            _right_dct_model.avgpool
        )
        self.face_dct_features = nn.Sequential(
            _face_dct_model.conv1,
            _face_dct_model.bn1,
            _face_dct_model.relu,
            _face_dct_model.maxpool,
            _face_dct_model.layer1,
            _face_dct_model.layer2,
            _face_dct_model.layer3,
            _face_dct_model.layer4,
            _face_dct_model.avgpool
        )
        
        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True
        for param in self.left_dct_features.parameters():
            param.requires_grad = True
        for param in self.right_dct_features.parameters():
            param.requires_grad = True
        for param in self.face_dct_features.parameters():
            param.requires_grad = True

        
        self.xl_dct = nn.Sequential(
            nn.Linear(fc_in_num, 512),
            nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )

        self.xr_dct = nn.Sequential(
            nn.Linear(fc_in_num, 512),
            nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        
        self.x_l = nn.Sequential(
            nn.Linear(fc_in_num, 512),
            nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        self.x_r = nn.Sequential(
            nn.Linear(fc_in_num, 512),
            nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )

        self.face = nn.Sequential(
            nn.Linear(fc_in_num, 512),
            nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        self.concat = nn.Sequential(
            nn.Linear(2560, 512),
            nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        self.fc = nn.Sequential(
            nn.Linear(514, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
            nn.Dropout(0.2)
        )

        
    def forward(self, x_in,  trained=True):
        face = x_in["face"]
        headlabel = x_in["headlabel"]
        left_eye = x_in["left_eye"]
        right_eye = x_in["right_eye"]
        face = x_in["face"]

        _,_, h, w = left_eye.size()
        mask = torch.ones((h, w), dtype=torch.int64, device = torch.device('cuda:0'))
        diagonal = w-2
        hf_mask = torch.fliplr(torch.triu(mask, diagonal)) != 1
        hf_mask = hf_mask.unsqueeze(0).expand(left_eye.size())

        ## left_eye_dct
        left_eye_dct = self.dct(left_eye)
        left_eye_dct = left_eye_dct * hf_mask
        left_eye_dct = self.idct(left_eye_dct)

        left_eye_dct = self.left_dct_features(left_eye_dct)
        left_eye_dct = torch.flatten(left_eye_dct, 1)
        left_eye_dct = self.xl_dct(left_eye_dct)

        ## left_eyendarr
        left_x = self.left_features(left_eye)
        left_x = torch.flatten(left_x, 1)
        left_x = self.x_l(left_x)
        # torch.Size([7, 1024])z
        
        # left_eye_dct
        right_eye_dct = self.dct(right_eye)
        right_eye_dct = right_eye_dct * hf_mask
        right_eye_dct = self.idct(right_eye_dct)

        right_eye_dct = self.left_dct_features(right_eye_dct)
        right_eye_dct = torch.flatten(right_eye_dct, 1)
        right_eye_dct = self.xr_dct(right_eye_dct)

        ## right eye
        right_x = self.right_features(right_eye)
        right_x = torch.flatten(right_x, 1)
        right_x = self.x_r(right_x)

        #face dct
        _,_, h, w = face.size()
        mask = torch.ones((h, w), dtype=torch.int64, device = torch.device('cuda:0'))
        diagonal = w-2
        hf_mask = torch.fliplr(torch.triu(mask, diagonal)) != 1
        hf_mask = hf_mask.unsqueeze(0).expand(face.size())

        face_dct = self.dct(face)
        face_dct = face_dct * hf_mask
        face_dct = self.idct(face_dct)

        # b,c,h,w = face_dct.size()
        
        face_dct = self.face_dct_features(face)
        face_dct = torch.flatten(face_dct, 1)
        face_dct = self.face(face_dct)

        features = torch.cat((left_x, left_eye_dct, right_x, right_eye_dct, face_dct ), dim=1)

        features = self.concat(features)
        # self.concat은 FC layer
        # torch.Size([7, 512])
        features_headpose = torch.cat((features, headlabel), dim=1)
        # torch.Size([7, 514])

        gaze = self.fc(features_headpose)
        # torch.Size([7, 2])

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


