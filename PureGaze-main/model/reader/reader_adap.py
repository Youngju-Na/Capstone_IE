import os
import cv2 
import torch
import random
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle
from glob import glob


def Decode_MPII(line):
    anno = edict()
    ##origin data preprocess
    # anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    # anno.name = line[3]

    # anno.gaze3d, anno.head3d = line[5], line[6]
    # anno.gaze2d, anno.head2d = line[7], line[8]

    ## constructed by js
    anno.ext = '.jpg'
    # anno.latents = torch.load('../dataset/latents/'+line[0]+'.pt', **{})
    anno.latents = []

    anno.face = line[0] + anno.ext
    anno.name = line[0]
    gaze2d = [line[1],line[2]]
    anno.gaze2d = ",".join(gaze2d)

    head2d = [line[3], line[4]]
    anno.head2d = ",".join(head2d)

    eye_coord = [line[5], line[6], line[7], line[8]]
    anno.eye_coord = ",".join(eye_coord)
    
    return anno

def Decode_Diap(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d, anno.head3d = line[4], line[5]
    anno.gaze2d, anno.head2d = line[6], line[7]
    return anno

def Decode_Gaze360(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d = line[4]
    anno.gaze2d = line[5]
    return anno

def Decode_ETH(line):
    anno = edict()
    anno.gaze2d = line[2]
    anno.head2d = line[1]
    anno.name = line[0]
    anno.ext = '.jpg'
    anno.face = line[0] + anno.ext
    return anno

def Decode_GazeCapture(line):
    anno = edict()
    anno.face = line[0]
    anno.gaze2d = line[1]
    anno.head2d = line[2]
    anno.name = line[3]
    return anno

def Decode_Dict():
    mapping = edict()
    mapping.mpiigaze = Decode_MPII
    mapping.eyediap = Decode_Diap
    mapping.gaze360 = Decode_Gaze360
    mapping.eth = Decode_ETH
    mapping.gazecapture = Decode_GazeCapture
    return mapping


def long_substr(str1, str2):
    substr = ''
    for i in range(len(str1)):
        for j in range(len(str1)-i+1):
            if j > len(substr) and (str1[i:i+j] in str2):
                substr = str1[i:i+j]
    return len(substr)


def Get_Decode(name):
    mapping = Decode_Dict()
    keys = list(mapping.keys())
    name = name.lower()
    score = [long_substr(name, i) for i in keys]
    key  = keys[score.index(max(score))]
    return mapping[key]
    

class commonloader(Dataset): 
  def __init__(self, dataset):

    # Read source data
    self.source = edict() 
    self.dataset = dataset
    # self.source.line = []
    self.source.latent = dataset.latent
    self.source.root = dataset.image
    self.source.decode = Get_Decode(dataset.name)
    self.filenames = sorted(
            glob(os.path.join(self.source.root, '*'))
        )

    with open(dataset.label, "rb") as f:
        label_dict = pickle.load(f)
      
    self.source.labels = label_dict
    # build transforms
    self.transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    
  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    # Read souce information
    filename = self.filenames[idx]
    img = cv2.imread(filename)

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))
    img = self.transforms(img)

    filename_key = filename.split('\\')[-1].split('.')[0]
    label = self.source.labels[filename_key]
    label = np.array(label).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)

    data = edict()
    data.face = img
    data.name = filename
    data.dataset_name = self.dataset.name
    # data.latents = torch.load(self.source.latent+'/'+filename_key+'.pt', **{})


    # if self.dataset.name == 'mpii':
    #   headlabel = np.array(anno.head2d.split(",")).astype("float")
    #   headlabel = torch.from_numpy(headlabel).type(torch.FloatTensor)
    #   data.headlabel = headlabel

    #   eye_coord = np.array(anno.eye_coord.split(",")).astype("int")
    #   eye_coord = torch.from_numpy(eye_coord).type(torch.FloatTensor)
    #   data.eye_coord = anno.eye_coord

    #   data.latents = anno.latents

    return data, label

def loader(source, batch_size, shuffle=False,  num_workers=0):
  dataset = commonloader(source)
  print(f"-- [Read Data]: Total num: {len(dataset)}")
  print(f"-- [Read Data]: Source: {source.label}")
  load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
  return load

if __name__ == "__main__":
  
  path = './p00.label'
# d = loader(path)
# print(len(d))
# (data, label) = d.__getitem__(0)

