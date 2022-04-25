import os, sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "/home/chengyihua/utils/")
import model
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
import ctools, gtools
from tqdm import tqdm


def main(train):
    
    # Setup-----------------------------------------------------------
    dataloader = importlib.import_module(f"reader.{config.reader}")

    torch.cuda.set_device(config.device)

    attentionmap = cv2.imread(config.map, 0)/255
    attentionmap = torch.from_numpy(attentionmap).type(torch.FloatTensor)

    data = config.data
    save = config.save
    params = config.params
    val_data = config.val_data


    # Prepare dataset-------------------------------------------------
    dataset = dataloader.loader(data, params.batch_size, shuffle=True, num_workers=0)

    # Build model
        # build model ------------------------------------------------
    print("===> Model building <===")
    net = model.Model(); net.train(); net.cuda()

    # if config.pretrain:
    #     net.load_state_dict(torch.load(config.pretrain), strict=False)

    print("optimizer building")
    geloss_op = model.Gelossop(attentionmap, w1=3, w2=1)
    deloss_op = model.Delossop()

    ge_optimizer = optim.Adam(net.feature.parameters(),
             lr=params.lr, betas=(0.9,0.95))

    ga_optimizer = optim.Adam(net.gazeEs.parameters(), 
             lr=params.lr, betas=(0.9,0.95))

    de_optimizer = optim.Adam(net.deconv.parameters(), 
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
            for i, (data, label) in enumerate(dataset):

                # Acquire data
                data["face"] = data["face"].cuda()
                label = label.cuda()
                # forward
                gaze, img = net(data)

                ge_optimizer.zero_grad()
                ga_optimizer.zero_grad()
                de_optimizer.zero_grad()

                for param in net.deconv.parameters():
                    param.requires_grad=False

                # loss calculation
                geloss = geloss_op(gaze, img, label, data["face"])
                geloss.backward(retain_graph=True)


                for param in net.deconv.parameters():
                    param.requires_grad=True


                for param in net.feature.parameters():
                    param.requires_grad = False

                deloss = deloss_op(img, data["face"])
                deloss.backward()
                for param in net.feature.parameters():
                    param.requires_grad=True
 
                ge_optimizer.step()
                ga_optimizer.step()
                de_optimizer.step()
                
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
                    # data["latents"] = data["latents"].cuda()

                    pre_gaze, _ = net(data)

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
 
