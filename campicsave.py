import argparse
import os
import cv2
import time
import torch
import datetime
import numpy as np
from torch import nn

from torchvision import models
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.ablation_cam_new import AblationCAMNEW
from pytorch_grad_cam.grad_cam_new import Grad_CAM_NEW
from pytorch_grad_cam.layer_cam_new import Layer_CAM_NEW
from pytorch_grad_cam.score_cam_new import ScoreCAMNew
from pytorch_grad_cam.mgrad_cam import MGradCAM

from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image, show_image

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSigmoidTarget




def get_cam(method, target_layers):
    cam_algorithm = methods[method]
    cam = cam_algorithm(model=model, target_layers=target_layers, use_cuda=use_cuda)
    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32
    return cam

def get_data(line):
    line = line.split()
    path = line[0]
    label = [int(i) for i in line[1].split(',')]
    return path, label

def save_cam(method, grayscale_cams, target, image_path):
    path = os.path.join(save_path, method)
    if not os.path.exists(path):
        os.makedirs(path)
    for tar, cam in zip(target, grayscale_cams):
        np.save(os.path.join(path, image_path.split('/')[-1].split('.')[0] + '_' + str(tar) + '.npy'), cam)
        
def process(method, target_layers, txt_path):
    with open(txt_path, encoding='utf-8') as f:
            lines = f.readlines()
    for line in tqdm(lines):
        image_path, target = get_data(line)
        rgb_img = cv2.imread(image_path)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, [224, 224])
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        targets = [ClassifierOutputTarget(int(tar)) for tar in target]
        input_tensor = torch.cat([input_tensor for tar in target], dim=0)
        
        cam = get_cam(method, target_layers)
        grayscale_cams = cam(input_tensor=input_tensor,
                                targets=targets,
                                aug_smooth=aug_smooth,
                                eigen_smooth=eigen_smooth)
        save_cam(method, grayscale_cams, target, image_path)
        # print(image_path)
        # break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam', type=str, default='gradcam')
    args = parser.parse_args()
    
    methods = {"gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "scorecamnew": ScoreCAMNew,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "ablationcamnew": AblationCAMNEW,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise,
        "layercamnew": Layer_CAM_NEW,
        "gradcamnew": Grad_CAM_NEW,
        "mgradcam": MGradCAM}

    home_dir = os.path.expanduser("~")
    save_path = os.path.join(home_dir, 'weights/CAM/campic')
    aug_smooth = False
    eigen_smooth = False

    use_cuda = True if torch.cuda.is_available()  else False
    device = torch.device("cuda") if torch.cuda.is_available()  else torch.device("cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, 20)
    resume = os.path.join(home_dir, r"weights/coco_cam/loss_20221102123343/best_acc.pth")
    model.load_state_dict(torch.load(resume, map_location='cpu')['model'])
    model = model.eval().to(device)
    
    process(args.cam, [model.layer4], os.path.join(home_dir, 'weights/coco_cam/train_with_box.txt'))
    process(args.cam, [model.layer4], os.path.join(home_dir, 'weights/coco_cam/val_with_box.txt'))
    
    # CUDA_VISIBLE_DEVICES=0 python ~/CAM/campicsave.py --cam gradcam