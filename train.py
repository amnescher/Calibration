import argparse
import copy
import json
import os.path
import random
from collections import deque
from pathlib import Path
import cv2 as cv
from matplotlib.transforms import Transform
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm
import matplotlib.pyplot as plt
from soccerpitch import SoccerPitch
from dataloader_rtsw import SoccerNetDataset
from network import SegmentationNetwork,load_ckp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import FocalLoss, IoULoss,DiceLoss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test')

    parser.add_argument('-s', '--dataset', default="", type=str,
                        help='Path to the SoccerNet-V3 dataset folder')
    parser.add_argument('--split', required=False, type=str, default="debug", help='Select the split of data')
    parser.add_argument('--epochs',required=False, type = int, default = 100,help='Number of epochs for training')
    parser.add_argument('--batchSize',required=False, type = int, default=32,help='Number of epochs for training')
    parser.add_argument('--transform',required=False, type = bool, default=True,help='Apply any data agumentaion')
    parser.add_argument('--resolution',required=False, type = dict, default={"width":512, "height":512},help='resolution of the image if resizing is applied')
    parser.add_argument('--checkPoint',required=False, type = bool, default=False, help='Loading from checkpoint')
    parser.add_argument('--mode',required=False, type = str, default="train", help='train/finetunning/test mode')
    parser.add_argument('--debugging',required=False, type = bool, default = False, help='train/finetunning/test mode')
    parser.add_argument('--scheduler_step',required=False, type = int, default= 15, help='learning rate scheduler step')
    parser.add_argument('--scheduler_gamma',required=False, type = float, default=0.1, help='gamma  coefficient for learning rate scheduler')
    parser.add_argument('--lr',required=False, type = float, default=0.001, help='learning rate')
    parser.add_argument('--loss_weight',required=False, type = float, default = 0.00008, help='Weight for the weighted sum of losses')
    parser.add_argument('--num_classes',required=False, type = int, default=21, help='Number of segmentation classes ')
    parser.add_argument('--mean_std_file',required=False, type = str, default="", help='mean and std of image data')
    parser.add_argument('--trained_model',required=False, type = str, default="", help='Path to pre-trained model')
    parser.add_argument('--ckp_dir',required=False, type = str, default="", help='Chekpoint Directory')


    args = parser.parse_args()
    writer = SummaryWriter()

    startEpoch = 0
    



    if args.mode =="finetunning":
        model_dir = os.path.join(args.trained_model, "soccer_pitch_segmentation.pth")
        if not os.path.exists(model_dir):
            print("Invalid dataset path to pre-trained model, pre-trained model is needed!")
            exit(-1)
    else:
        model_dir = None


    std_path =  os.path.join(args.mean_std_file, "std.npy")
    mean_path =  os.path.join(args.mean_std_file, "mean.npy")
    if not os.path.exists(mean_path):
        print("Invalid data path !")
        exit(-1)

    print("Loading Dataset")

    dataset_train = SoccerNetDataset(args.dataset,split = "train", transform=args.transform, width=args.resolution["width"], height=args.resolution["height"])
    dataset_eval = SoccerNetDataset(args.dataset,split = "valid", transform = args.transform, width=args.resolution["width"], height=args.resolution["height"])
    dataloader_train = torch.utils.data.DataLoader(dataset_train,batch_size = args.batchSize, drop_last =True,num_workers = 8)
    dataloader_eval = torch.utils.data.DataLoader(dataset_eval,batch_size = args.batchSize, drop_last =True, num_workers = 8)
    if args.debugging:
        print("--------- Debugging Mode ----------")
        dataset_train =  torch.utils.data.Subset(dataset_train, [1,5000])
        dataset_eval =  torch.utils.data.Subset(dataset_eval, [1,164])
        dataloader_train = torch.utils.data.DataLoader(dataset_train,batch_size = args.batchSize, drop_last =False,num_workers =2)
        dataloader_eval = torch.utils.data.DataLoader(dataset_eval,batch_size = args.batchSize, drop_last =False, num_workers =2)
        
    
    
    model = SegmentationNetwork(model_file = model_dir,mode=args.mode,num_classes=args.num_classes)
    
    #optimizer needs to be better written
    optimizer = torch.optim.Adam(model.model.parameters(), lr =args.lr)
    
    
    criterion = (FocalLoss(),DiceLoss())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,15,30], gamma=args.scheduler_gamma)
    
    
    if args.mode == "train" and args.checkPoint:
        print("---Loading checkpoint-----")
        model,optimizer,startEpoch = load_ckp(args.ckp_dir,model,optimizer)

    for ech in range(startEpoch,args.epochs):
            
            
            train_loss, train_focal, train_dice = model.train_one_epoch(dataloader_train,optimizer,criterion,args.loss_weight,ech,args.ckp_dir, writer)
            eval_loss, eval_focal, eval_dice = model.eval_one_epoch(dataloader_eval,criterion,args.loss_weight,ech,writer)

            scheduler.step()

            #Writing training info into tensor board
            writer.add_scalars("Loss",{"train":train_loss,"Evla":eval_loss},ech)
            writer.add_scalars("Focal",{"train":train_focal,"Evla":eval_focal},ech)
            writer.add_scalars("Dice",{"train":train_dice,"Evla":eval_dice},ech)
            #Printing training info in console
            print(f"Epoch[{ech}/{args.epochs}]","TL: ", train_loss.cpu().detach().numpy(),
            " TFL:",train_focal.item()," TDL:",train_dice.item()
            ,"- - EL: ",eval_loss.cpu().detach().numpy()," EFL:",eval_focal.item()," EDL:",eval_dice.item())

    print("I'm writing into TensorBoard")
    writer.close()
    print("---Done!---")

    
