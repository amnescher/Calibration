
from PIL import Image
import cv2 as cv
from soccerpitch import SoccerPitch
import numpy as np
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
SMOOTH = 1e-6

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss




def IoULoss(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    
    #outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    print("------After--------")
    print("output shape---->", outputs.size())
    print("label size------>",labels.size())
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded 

#export
class DiceLoss:
    "Dice loss for segmentation"
    def __init__(self, axis=1, smooth=1e-6, reduction="sum", square_in_union=False):
        self.reduction = reduction
        self.square_in_union = square_in_union
        self.smooth = smooth
        self.axis = axis
        
    def __call__(self, pred, targ):
            targ = self._one_hot(targ, pred.shape[self.axis])
            #pred, targ = TensorBase(pred), TensorBase(targ)
            assert pred.shape == targ.shape, 'input and target dimensions differ, DiceLoss expects non one-hot targs'
            pred = self.activation(pred)
            sum_dims = list(range(2, len(pred.shape)))
            inter = torch.sum(pred*targ, dim=sum_dims)        
            union = (torch.sum(pred**2+targ, dim=sum_dims) if self.square_in_union
                else torch.sum(pred+targ, dim=sum_dims))
            dice_score = (2. * inter + self.smooth)/(union + self.smooth)
            loss = 1- dice_score
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
            return loss
    @staticmethod
    def _one_hot(x, classes, axis=1):
        "Creates one binay mask per class"
        return torch.stack([torch.where(x==c, 1, 0) for c in range(classes)], axis=axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)
    def decodes(self, x):    return x.argmax(dim=self.axis)

def generate_mask_annotation(image,extremities):
   

    mask_extr = np.zeros(image.shape[:-1], dtype=np.uint8)
    for class_number, class_ in enumerate(SoccerPitch.lines_classes):
        if class_ in extremities.keys():
            key = class_
            line = extremities[key]
            prev_point = line[0]
            for i in range(1, len(line)):
                next_point = line[i]
                cv.line(mask_extr,
                        (int(prev_point["x"] * mask_extr.shape[1]), int(prev_point["y"] * mask_extr.shape[0])),
                        (int(next_point["x"] * mask_extr.shape[1]), int(next_point["y"] * mask_extr.shape[0])),
                    class_number+1,
                        2)
                prev_point = next_point
    mask_extr = Image.fromarray(mask_extr.astype(np.uint8)).convert('P')
    return mask_extr


def get_concat_h(seg, det_lines, img, ann_lines=None):

    
    overlayed_mask = (cv.add(np.array(img), np.array(det_lines))).astype(np.uint8)
    overlayed_mask = Image.fromarray(overlayed_mask)
    dst = Image.new('RGB', (seg.width + det_lines.width+img.width, img.height*2))
    if ann_lines:
        overlayed_ann_mask = (cv.add(np.array(img), np.array(ann_lines))).astype(np.uint8)
        dst.paste(Image.fromarray(overlayed_ann_mask), (seg.width, img.height))
        dst.paste(ann_lines, (seg.width+det_lines.width, seg.height))

    dst.paste(img, (0, 0))
    dst.paste(seg, (seg.width, 0))
    dst.paste(det_lines, (seg.width+det_lines.width,0))
    dst.paste(overlayed_mask, (0, seg.height))
    
    
    return dst



def generate_mask(image,seg_pred,extremities,lines_palette,ann_mask=None):
    ''' Visualizing segmentation map. Using predicted extremities, lines are visualized
        as a mask and visualized alongside original image and semantic segmentation mask.

    :param image: image performed segmentation on
    :param seg_prediction: segmentation prediction tensor
    :param extremities: dictionary of predicted extrimies
    :param lines_palette:  color palette for predicted lines
    :return pred_mask: concatinated image, segmentation results and predicted lines.
    '''

    mask = Image.fromarray(seg_pred.astype(np.uint8)).convert('P')
    mask.putpalette(lines_palette)

    img_arr = np.asarray(image)
    mask_extr = np.zeros(img_arr.shape[:-1], dtype=np.uint8)
    for class_number, class_ in enumerate(SoccerPitch.lines_classes):
        if class_ in extremities.keys():
            key = class_
            line = extremities[key]
            prev_point = line[0]
            for i in range(1, len(line)):
                next_point = line[i]
                cv.line(mask_extr,
                        (int(prev_point["x"] * mask_extr.shape[1]), int(prev_point["y"] * mask_extr.shape[0])),
                        (int(next_point["x"] * mask_extr.shape[1]), int(next_point["y"] * mask_extr.shape[0])),
                    class_number+1,
                        2)
                prev_point = next_point
    mask_extr = Image.fromarray(mask_extr.astype(np.uint8)).convert('P')
    mask_extr.putpalette(lines_palette)
    
    annotation_mask = None
    if ann_mask:
        ann_mask.putpalette(lines_palette)
        annotation_mask = ann_mask.convert('RGB')
    pred_mask = get_concat_h(mask.convert('RGB'),mask_extr.convert('RGB'),image,annotation_mask)
    return pred_mask

def testSet_loader(testset_path):
    image_dir = os.path.join(testset_path, "rgb")
    ann_dir = os.path.join(testset_path, "label")
    # read iamge and mask from test set 
    
    frames = [f for f in os.listdir(image_dir) if ".png.png" in f]
    data = []
    for frame in frames:

            frame_index = frame.split(".png.png")[0]
            annotation_file = os.path.join(ann_dir, f"{frame_index}.png.png")
            
            
            if not os.path.exists(annotation_file):
                print( "*****Following annotation file doesn't exist**********")
                print(annotation_file)
                break
            img_path = os.path.join(image_dir, frame)  
            if annotation_file:
                data.append({
                    "image_path": img_path,
                    "annotations": annotation_file, })
    return data