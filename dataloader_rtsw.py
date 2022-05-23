"""
DataLoader used to train the segmentation network used for the prediction of extremities.
"""

import json
import os
import time
from argparse import ArgumentParser

import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from soccerpitch import SoccerPitch
from torchvision import datasets, transforms
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision 
import torch 
import random
class SoccerNetDataset(Dataset):
    def __init__(self,
                 datasetpath,
                 split="debug",
                 transform = False,
                 width=1920,
                 height=1080):
       
        self.width = width
        self.height = height
        self.trans = transform
        #Define some image transformation
        self.image_resize = T.Compose([
                                T.Resize((self.height,self.width)),
                                ])
        self.mask_resize = T.Compose([
                                T.Resize((self.height,self.width),interpolation = torchvision.transforms.InterpolationMode.NEAREST)
                                ])

        self.GaussianBlur = torchvision.transforms.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 0.2))
        self.RandomAutocontrast = torchvision.transforms.RandomAutocontrast(p=0.5)

        datasetpath = os.path.join(datasetpath, split)
        image_dir = os.path.join(datasetpath, "rgb")
        ann_dir = os.path.join(datasetpath, "label")
        if not os.path.exists(image_dir):
            print("Invalid path to images !")
            exit(-1)
        frame_list = os.listdir(image_dir)
        frames = [f for f in os.listdir(image_dir) if ".png.png" in f]

        self.data = []
        self.n_samples = 0
        #Color codes
        self.lines_palette = [0, 0, 0]
        for line_class in SoccerPitch.lines_classes:
            self.lines_palette.extend(SoccerPitch.palette[line_class])
        
        for frame in frames:

            frame_index = frame.split(".png.png")[0]
            annotation_file = os.path.join(ann_dir, f"{frame_index}.png.png")
            
            
            if not os.path.exists(annotation_file):
                print( "*****Following annotation file doesn't exist**********")
                print(annotation_file)
                break
            img_path = os.path.join(image_dir, frame)  
            if annotation_file:
                self.data.append({
                    "image_path": img_path,
                    "annotations": annotation_file,
                })
    def transform(self, image, mask):
        
        # Random crop
        if random.random()>0.5:
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(512, 512))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

        if random.random()> 0.5:
            angle = random.randint(-30, 30)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        if random.random()>0.5:
            image = self.GaussianBlur(image)

        image = self.RandomAutocontrast(image)

        return image, mask

    def __len__(self):
        return len(self.data)
    
    def improve_mask(self, mask):

        mask1 = np.asarray(mask)
        np_mask1 = np.zeros((21,self.height,self.width))
        for i in range(21):       
            np_mask1[i,:,:] = (mask1== i)
        kernel = np.ones((4,4), np.uint8)

        lines = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

        for l in lines:
            np_mask1[l,:,:] = cv.dilate(np_mask1[l,:,:],kernel,iterations=1)
            np_mask1[l,:,:] = np.where(np_mask1[l,:,:] == 1, l, 0)
            
        final = np.amax(np_mask1, axis=0)
        return final

    def __getitem__(self, index):

        item = self.data[index]
        
        
        img = Image.open(item["image_path"])
        mask = Image.open(item["annotations"]).convert('L') 

        if self.trans:
            img = self.image_resize (img)
            mask = self.mask_resize(mask) 
        
        img,mask = self.transform(img,mask)
            
        
        mask = self.improve_mask(mask)
        mask = torch.tensor(np.asarray(mask), dtype=torch.float32)
        img = T.ToTensor()(img)
        
        # TO visualize masks and save them in a directorry uncomment below lines
        #mask.putpalette(self.lines_palette)
        #mask.convert('RGB')
        #mask.save((item["annotations"]).split("label")[0]+"Half_size_label"+(item["annotations"]).split("label")[1])


        return img, mask