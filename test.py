import argparse
import copy
import json
import os.path
import random
from collections import deque
from pathlib import Path
import torchvision
import cv2 as cv
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm
import glob
from soccerpitch import SoccerPitch
from temp_network import SegmentationNetwork, generate_class_synthesis, get_line_extremities
from utils import get_concat_h,generate_mask,generate_mask_annotation, testSet_loader
import torch
import shutil
import torchvision.transforms as T
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test')

    parser.add_argument('-s', '--soccernet', default="", type=str,
                        help='Path to the SoccerNet-V3 dataset folder')

    parser.add_argument('-p', '--prediction', default="./results_bis", required=False, type=str,
                        help="Path to the prediction folder")

    parser.add_argument('--mask_prediction', default="./results_masks", required=False, type=str,
                        help="Path to the prediction folder")
    parser.add_argument('--split', required=False, type=str, default="test", help='Select the split of data')
    parser.add_argument('--masks', required=False, type=bool, default=True, help='Save masks in prediction directory')
    parser.add_argument('--ann_test', required=False, type=bool, default=True, help='If test set comes with annotation files')
    parser.add_argument('--trans', required=False, type=bool, default=True, help='perform data agumenatation on images')
    parser.add_argument('--resolution_width', required=False, type=int, default=512,
                        help='width resolution of the images')
    parser.add_argument('--resolution_height', required=False, type=int, default=512,
                        help='height resolution of the images')
    args = parser.parse_args()

    lines_palette = [0, 0, 0]
    for line_class in SoccerPitch.lines_classes:
        lines_palette.extend(SoccerPitch.palette[line_class])
    json_path = ''
    calib_net = SegmentationNetwork(model_file = "checkpoints/checkpoint_34.pth",mode = "test",num_classes=21)

    dataset_dir = os.path.join(args.soccernet, args.split)
    
    image_transform = T.Compose([
                                T.Resize((args.resolution_height,args.resolution_width)),
                                ])
    mask_transform = T.Compose([
                                T.Resize((args.resolution_height,args.resolution_width),interpolation = torchvision.transforms.InterpolationMode.NEAREST)
                                ])

    output_prediction_folder = os.path.join(args.prediction, args.split)
    if not os.path.exists(output_prediction_folder):
                os.makedirs(output_prediction_folder)
    data = testSet_loader(args.soccernet)
    #perform detection on all test samples
    with tqdm(enumerate(data), total=len(data) , ncols=160) as t:
        for i, frame in t:
            
            img = Image.open(data[i]["image_path"])
            if args.trans:
                image = image_transform(img)
            
            #perform prediction
            semlines = calib_net.analyse_image(T.ToTensor()(image))

            mask = None
            skeletons = generate_class_synthesis(semlines, 6)
            extremities = get_line_extremities(skeletons, 40, args.resolution_width, args.resolution_height)
            prediction = extremities
            mask_name = (data[i]["image_path"].split("rgb/")[1]).split('.png.png')[0]
            prediction_file = os.path.join(json_path, f"extremities_{mask_name}.json")

            with open(prediction_file, "w") as f:
                json.dump(prediction, f, indent=4)
                
            ann_mask = None
            if args.masks:
                mask_file = os.path.join(args.mask_prediction, data[i]["image_path"].split("rgb/")[1])
                if args.ann_test:
                    ann_mask = Image.open(data[i]["annotations"]).convert('L') 
                    ann_mask.putpalette(lines_palette)
                    ann_mask.convert('RGB')
                    if args.trans:
                         ann_mask = mask_transform(ann_mask)
                pred_mask = generate_mask(image,semlines,extremities,lines_palette, ann_mask = ann_mask )
                pred_mask.save(mask_file)
            
          
        
        
