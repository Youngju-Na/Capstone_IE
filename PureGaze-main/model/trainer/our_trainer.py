import os, sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "/home/chengyihua/utils/")
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import yaml
import cv2
import ctools
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import argparse
from torchvision.utils import save_image
import our_model
import rt_gene
import latents_gaze
from tqdm import tqdm
import ctools, gtools

from tester import our_total
def cropEyes(face_img, eye_coord):
    left_eyes = []
    right_eyes = []
    
    x = 75
    y = 36
    
    for i in range(face_img.size()[0]):
        coord = eye_coord[i].split(',')
        l_x = int(coord[0])
        l_y = int(coord[1])
        r_x = int(coord[2])
        r_y = int(coord[3])
        left_eye = face_img[i,:, l_y-y: l_y+y, l_x-x : l_x+x]
        left_eyes.append(left_eye.cuda())

        right_eye = face_img[i,:, r_y-y : r_y+y, r_x-x : r_x+x]
        right_eyes.append(right_eye.cuda())

    left_eyes = torch.cat([x.unsqueeze(0) for x in left_eyes], dim=0)
    right_eyes = torch.cat([x.unsqueeze(0) for x in right_eyes], dim=0)

    return left_eyes, right_eyes

def main(train):
    
    # Setup-----------------------------------------------------------
    dataloader = importlib.import_module(f"reader.{config.reader}")
    torch.cuda.set_device(config.device)

    data = config.data
    val_data = config.val_data
    save = config.save
    params = config.params


    # Prepare dataset-------------------------------------------------
    dataset = dataloader.loader(data, params.batch_size, shuffle=False, num_workers=0)


    # Build model
        # build model ------------------------------------------------
    print("===> Model building <===")
    net = latents_gaze.Model(); net.train(); net.cuda()
    # net = rt_gene.Model(); net.train(); net.cuda()
    

    if config.pretrain:
        net.load_state_dict(torch.load(config.pretrain), strict=False)

    print("optimizer building")
    geloss_op = latents_gaze.Gelossop()

    ge_optimizer = optim.Adam(net.parameters(),
             lr=params.lr, betas=(0.9,0.95))

    # scheduler = optim.lr_scheduler.StepLR(optimizer,
            #step_size=params.decay_step, gamma=params.decay)

    # prepare for training ------------------------------------

    length = len(dataset);
    total = length * params.epoch

    savepath = os.path.join(save.metapath, save.folder, f"checkpoint")

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    timer = ctools.TimeCounter(total)

  
    print("Training")
    with open(os.path.join(savepath, "train_log"), 'w') as outfile:
        for epoch in range(1, config["params"]["epoch"]+1):
            for i, (data, label_gt) in enumerate(dataset):

                # Acquire data
                data["face"] = data["face"].cuda()

                if data["dataset_name"][0] == 'mpii':
                    data["headlabel"] = data["headlabel"].cuda()
                data["latents"] = data["latents"].cuda()

                    ##eye crop
                    # left_eye, right_eye = cropEyes(data["face"], data['eye_coord'])
                    # data["left_eye"] = left_eye
                    # data["right_eye"] = right_eye

                label_gt = label_gt.cuda()
 
                # forward
                pre_gaze = net(data)

                ge_optimizer.zero_grad()

                # loss calculation
                geloss = geloss_op(pre_gaze, label_gt)
                geloss.backward(retain_graph=True)

                ge_optimizer.step()
                
                rest = timer.step()/3600

                # print logs
                if i % 2500 == 0:
                    log = f"[{epoch}/{params.epoch}]: " + \
                          f"[{i}/{length}] " +\
                          f"gloss:{geloss} " +\
                          f"lr:{params.lr} " +\
                          f"rest time:{rest:.2f}h"

                    print(log)
                    outfile.write(log + "\n")
                    sys.stdout.flush()   
                    outfile.flush()

            if epoch % config["save"]["step"] == 0:
                net.eval() 
                accs = 0; count = 0

                val_dataset = dataloader.loader(val_data, 1, shuffle=False, num_workers=0)
                tqdm_test = tqdm(val_dataset, ncols=80)
                for i, (data, val_label_gt) in enumerate(tqdm_test):

                    data["face"] = data["face"].cuda()
                    data["latents"] = data["latents"].cuda()

                    pre_gaze = net(data)

                    val_gt = val_label_gt[0].cpu().numpy()
                    result = pre_gaze[0].cpu().detach().numpy()

                    accs += gtools.angular(gtools.gazeto3d(val_gt),
                            gtools.gazeto3d(result))
                    count += 1

                avg = accs/count
                log = f"Total Num: {count}, avg: {avg}"

                curr_avg = avg
                if epoch == 1:
                    ex_avg = curr_avg
                    best_epoch = 1
                    
                if curr_avg < ex_avg:
                    ex_evg = curr_avg
                    best_epoch = epoch
                    print(f'best_{best_epoch}')
                    torch.save(net.state_dict(), os.path.join(savepath, f"Iter_best.pt"))
                print(log)
                outfile.write(log + "\n")
                sys.stdout.flush()   
                outfile.flush()
                torch.save(net.state_dict(), os.path.join(savepath, f"Iter_{epoch}_{save.name}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')

    parser.add_argument('-c', '--config', type=str,
                        help='Path to the config file.')

    args = parser.parse_args()

    config = edict(yaml.load(open(args.config), Loader=yaml.FullLoader))

    main(config)
 
