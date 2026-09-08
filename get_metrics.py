import argparse
import math
import tracemalloc
from multiprocessing import Process
import time
import datetime
from typing import List

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import gc
import _thread
from torch.utils.data.dataset import T_co
from tqdm import tqdm
from torchvision.transforms import Normalize
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
    FinerCAM, \
    KPCACAM, \
    PCAMpm, \
    GradCAMElementWise
from pytorch_grad_cam.ablation_cam_new import AblationCAMNEW
from pytorch_grad_cam.layer_cam_new import Layer_CAM_NEW
from pytorch_grad_cam.grad_cam_new import Grad_CAM_NEW
from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange
from pytorch_grad_cam.mgrad_cam import MGradCAM
from pytorch_grad_cam.score_cam_new import ScoreCAMNew
from pytorch_grad_cam.utils import deprocess_image
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget, ClassifierOutputSigmoidTarget, \
    ClassifierOutputTarget
from pytorch_grad_cam.metrics.road import ROADCombined, ROADLeastRelevantFirstAverage, ROADMostRelevantFirstAverage
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst
import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class CAM_Dataset(Dataset):
    def __init__(self, annotation_lines, w, h):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.w = w
        self.h = h

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        line = self.annotation_lines[index].split()
        image_raw = cv2.imread(line[0])
        w_raw, h_raw = image_raw.shape[1], image_raw.shape[0]
        image_raw = cv2.resize(image_raw, [self.w, self.h])
        image_raw = image_raw[:, :, ::-1]
        rgb_img = np.float32(image_raw) / 255
        image = self.trans(rgb_img)[0]
        image_id = line[0].split('/')[-1]
        label = [int(i) for i in line[1].split(',')]
        boxs = self.get_box(line[4:], w_raw, h_raw)
        return image, rgb_img, label, image_id, boxs

    @staticmethod
    def trans(rgb_img):
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        return input_tensor

    def get_box(self, boxs_str, w, h):
        boxs = np.zeros([len(boxs_str), 5])
        for index, box_str in enumerate(boxs_str):
            boxs_ = np.array([int(x) for x in box_str.split(",")])
            boxs_[0], boxs_[1] = (self.w / w) * boxs_[0], (self.w / w) * boxs_[1]
            boxs_[2], boxs_[3] = (self.h / h) * boxs_[2], (self.h / h) * boxs_[3]
            boxs[index, :] = boxs_
        return boxs

    @staticmethod
    def collate_fn(batch):
        image, rgb_img, label, image_id, boxs = list(zip(*batch))
        label_list, image_list, image_id_list, rgb_img_list, boxs_list = [], [], [], [], []
        for index, lab in enumerate(label):
            for l in lab:
                label_list.append(l)
                image_list.append(image[index])
                image_id_list.append(image_id[index])
                rgb_img_list.append(rgb_img[index][np.newaxis, :])
                boxs_list.append(boxs[index][np.where(boxs[index][:, -1] == l), :][0, ..., :-1])
        return torch.stack(image_list, dim=0), np.concatenate(rgb_img_list,
                                                              axis=0), label_list, image_id_list, boxs_list


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=1, help="num_workers")
    parser.add_argument('--method', type=str, default='mgradcam')
    parser.add_argument('--path', type=str, default="/devdata/home/homefun/")
    parser.add_argument('--savepath', type=str, default="/devdata/home/homefun/weights/CAM")
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--normalization_method', type=str, default="", help="_max_min, _softmax, _Z-score")
    parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--layer', type=int, default=432)
    args = parser.parse_args()
    return args


def get_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, 20)
    if resume != "":
        model.load_state_dict(torch.load(resume, map_location='cpu')['model'])
    return model.eval()


def show_cam(rgb_img, grayscale_cam):
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    plt.Figure()
    plt.imshow(cam_image)
    plt.show()


def show_pic(cam_image):
    plt.Figure()
    plt.imshow(cam_image)
    plt.show()


def concat(save_path):
    path_list = []
    for p in os.listdir(save_path):
        if p[-3:] == 'csv':
            path_list.append(os.path.join(save_path, p))
    if len(path_list) != 1:
        csv = pd.DataFrame()
        for p in path_list:
            try:
                csv_ = pd.read_csv(p)
                csv_.index = list(range(len(csv_)))
                csv = pd.concat([csv, csv_], ignore_index=True)
                csv.index = list(range(len(csv)))
            except Exception:
                pass
        csv.to_csv(os.path.join(save_path, 'record.csv'), index=False)


def get_hits_per_image_label(grayscale_cams, labels, boxs, rgb_images):
    assert grayscale_cams.shape[0] == len(labels)
    grayscale_cams = torch.from_numpy(grayscale_cams).cuda()

    # # 画图
    # cam_image_list = [show_cam_on_image(rgb_images[i], grayscale_cams[i].cpu().numpy(), use_rgb=True)
    #                   for i in range(grayscale_cams.shape[0])]

    #     二值化cams
    cam_list = []
    for index in range(grayscale_cams.shape[0]):
        cam = grayscale_cams[index]
        cam[cam < torch.mean(cam)] = 0
        cam[cam != 0] = 1
        cam_list.append(cam)
    cams = torch.stack(cam_list, dim=0).cuda()

    label_cam_list = []
    for index, box in enumerate(boxs):
        cam_label = torch.ones_like(cam_list[0]) * 2
        for b in box:
            cam_label[int(b[2]):int(b[3]) + 1, int(b[0]):int(b[1]) + 1] = 1

            # # 画图
            # cv2.rectangle(cam_image_list[index], (int(b[0]), int(b[2])),
            #               (int(b[1]), int(b[3])), (0, 0, 255), 2)

        label_cam_list.append(cam_label)

    # # 画图
    # for index, cam_image in enumerate(cam_image_list):
    #     plt.subplot(1, len(cam_image_list), index + 1)
    #     plt.imshow(cam_image)
    # plt.tight_layout()
    # plt.savefig('./pic/metrics.jpg')

    label_cams = torch.stack(label_cam_list, dim=0).cuda()

    result = cams * label_cams
    result_true = torch.sum(
        torch.where(result == 1, torch.ones_like(result), torch.zeros_like(result)).reshape(result.shape[0], -1),
        dim=1)
    result_false = torch.sum(
        torch.where(result == 2, torch.ones_like(result), torch.zeros_like(result)).reshape(result.shape[0], -1),
        dim=1)
    result = result_true / (result_true + result_false + 1e-7)
    return result


def get_loc_per_image_label(grayscale_cams, labels, boxs):
    assert grayscale_cams.shape[0] == len(labels)
    grayscale_cams = torch.from_numpy(grayscale_cams).cuda()
    label_cam_list = []
    for index, box in enumerate(boxs):
        cam_label = torch.zeros_like(grayscale_cams[0])
        for b in box:
            cam_label[int(b[2]):int(b[3]) + 1, int(b[0]):int(b[1]) + 1] = 1

        label_cam_list.append(cam_label)

    label_cams = torch.stack(label_cam_list, dim=0).cuda()

    result = grayscale_cams * label_cams

    result = [(torch.sum(re) / (torch.sum(cam) + 1e-7)).cpu().numpy() for re, cam in zip(result, grayscale_cams)]
    return result


def get_drop_increase_per_image(grayscale_cams, device, targets, model, rgb_imgs):
    rgb_imgs = torch.from_numpy(rgb_imgs)
    grayscale_cams = torch.from_numpy(grayscale_cams)
    threshold_list = [np.percentile(cams, 50) for cams in grayscale_cams]
    drop_img_list = [get_drop_data(rgb_imgs[j], grayscale_cams[j], threshold_list[j], 0) for j in range(len(rgb_imgs))]
    drop_tensor = torch.cat(drop_img_list, dim=0)
    with torch.no_grad():
        raw_outputs = model(normalizer(torch.permute(rgb_imgs, [0, 3, 1, 2])).to(device)).detach().cpu()
        now_outputs = model(drop_tensor.to(device)).detach().cpu()
        drop_increase_list = np.array([(target(raw_output).numpy() -
                                        target(now_output).numpy()) / target(raw_output).numpy()
                                       for target, now_output, raw_output in zip(targets, now_outputs, raw_outputs)])
    return drop_increase_list


def get_drop_data(rgb_img, cam, threshold, delete=1):
    cam = cam.repeat(3, 1, 1)
    rgb_img = torch.permute(rgb_img, [2, 0, 1])
    if delete:
        drop_data = torch.where(cam >= threshold,
                                torch.ones_like(rgb_img) * 0.5,
                                rgb_img)
    else:
        drop_data = torch.where(cam >= threshold,
                                rgb_img,
                                torch.ones_like(rgb_img) * 0.5)
    # drop_data = rgb_img
    drop_data = torch.unsqueeze(drop_data, dim=0)
    drop_data = normalizer(drop_data)

    # 验证加灰度
    # drop_data = drop_data.cpu().numpy()
    return drop_data


def get_auc_score(model, device, rgb_imgs, grayscale_cams, targets, percentiles, batch_size, delete=0):
    threshold_list = [[np.percentile(cams, per) for per in percentiles] for cams in grayscale_cams]
    drop_img_list = [[get_drop_data(rgb_imgs[j],
                                    grayscale_cams[j], th, delete=delete) for th in
                      threshold_list[j]] for j in range(len(threshold_list))]
    drop_tensor = torch.cat([torch.cat(drop_img, dim=0) for drop_img in drop_img_list], dim=0)
    output_list = []
    with torch.no_grad():
        for i in range(math.ceil(drop_tensor.shape[0] / batch_size)):
            x = drop_tensor[batch_size * i:
                            batch_size * (i + 1) if
                            batch_size * (i + 1) < drop_tensor.shape[0] else drop_tensor.shape[0], ...]
            x = x.to(device)
            output_list.append(model(x).detach().cpu())
        output_list = torch.cat(output_list, dim=0)
        output_list = [output_list[j * len(drop_img_list[j]):
                                   (j + 1) * len(drop_img_list[j]), ...] for j in range(len(drop_img_list))]
        scores = np.array([target(output).numpy() for output, target in zip(output_list, targets)])
    return scores


def get_AUC_per_image_label_(grayscale_cams, device, targets, model, rgb_imgs, batch_size,
                             percentiles_delete, percentiles_insert):
    rgb_imgs = torch.from_numpy(rgb_imgs)
    grayscale_cams = torch.from_numpy(grayscale_cams)
    scores_delete = get_auc_score(model, device, rgb_imgs, grayscale_cams, targets,
                                  percentiles_delete, batch_size, delete=1)
    scores_insert = get_auc_score(model, device, rgb_imgs, grayscale_cams, targets,
                                  percentiles_insert, batch_size, delete=0)
    # threshold_list = [[np.percentile(cams, per) for per in percentiles_delete] for cams in grayscale_cams]
    #
    # assert len(threshold_list) == len(grayscale_cams) == len(rgb_imgs)
    # grayscale_cams = torch.from_numpy(grayscale_cams)
    # drop_img_list = [[get_drop_data(rgb_imgs[j],
    #                                 grayscale_cams[j], th, device, delete=1) for th in
    #                   threshold_list[j]] for j in range(len(threshold_list))]
    # # 验证加灰度
    # # for img in drop_img_list[1]:
    # #     plt.figure()
    # #     plt.imshow(img)
    # #     plt.show()
    # drop_tensor_list = [torch.cat(drop_img, dim=0) for drop_img in drop_img_list]
    # output_list = [model(drop_tensor) for drop_tensor in drop_tensor_list]
    # scores_delete = np.array([target(output).detach().cpu().numpy() for output, target in zip(output_list, targets)])

    # cam_metric_insert = ROADLeastRelevantFirstAverage(return_diff=False, percentiles=percentiles)
    # cam_metric_delete = ROADMostRelevantFirstAverage(return_diff=False, percentiles=percentiles[::-1])
    # scores_insert = cam_metric_insert(input_tensor, grayscale_cams, targets, model)
    # scores_delete = cam_metric_delete(input_tensor, grayscale_cams, targets, model)
    return scores_insert, scores_delete


def record(record_csv, labels, image_ids, logits, insert, delete, hits, loc, drop_increase_list,
           percentiles_delete, percentiles_insert):
    insert_ = np.mean(insert, axis=1)
    delete_ = np.mean(delete, axis=1)

    for index, (label) in enumerate(labels):
        new_column = pd.Series(dtype='float64')
        new_column["image_id"] = image_ids[index]
        new_column["label"] = label
        new_column["logits"] = logits[index, label]
        new_column["drop"] = drop_increase_list[index] if drop_increase_list[index] > 0 else 0
        new_column["increase"] = 1 if drop_increase_list[index] < 0 else 0
        for item in range(len(percentiles_delete)):
            new_column["delete_{}".format(percentiles_delete[item])] = delete[index][item]
        for item in range(len(percentiles_insert)):
            new_column["insert_{}".format(percentiles_insert[item])] = insert[index][item]
        new_column["insert_mean"] = insert_[index]
        new_column["delete_mean"] = delete_[index]
        new_column["hits"] = hits[index]
        new_column["loc"] = loc[index]
        record_csv = pd.concat([record_csv, pd.DataFrame(new_column).T], ignore_index=True)
    return record_csv, np.sum(insert_) / len(labels), np.sum(delete_) / len(labels)


class start():
    def __init__(self, start_: int, end_: int, csv_save_path: str,
                 model, cam, percentiles_delete, percentiles_insert,
                 w: int = 224,
                 h: int = 224,
                 batch_size: int = 1,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False,
                 num_workers: bool = 1,
                 val_txt: str = r"/devdata/home/homefun/weights/coco_cam/val_with_box.txt"):
        self.percentiles_insert = percentiles_insert
        self.percentiles_delete = percentiles_delete
        self.model = model
        self.end_ = end_
        self.start_ = start_
        self.csv_save_path = csv_save_path
        self.cam = cam
        self.w = w
        self.h = h
        self.batch_size = batch_size
        self.aug_smooth = aug_smooth
        self.eigen_smooth = eigen_smooth
        self.num_workers = num_workers
        self.val_txt = val_txt

    def run(self):
        start_time = time.time()

        # set device
        torch.cuda.set_device(1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # dataloader
        with open(self.val_txt, encoding='utf-8') as f:
            val_lines = f.readlines()
        val_lines = val_lines[self.start_:self.end_ if self.end_ <= len(val_lines) else len(val_lines)]
        val_dataset = CAM_Dataset(val_lines, self.w, self.h)
        gen_val = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                             collate_fn=val_dataset.collate_fn)

        # record metric
        record_csv = pd.DataFrame()

        # get metrics
        insert_ = 0
        delete_ = 0
        hits_ = 0
        loc_ = 0
        with tqdm(total=len(gen_val)) as pbar:
            for index, (images, rgb_images, labels, image_ids, boxs) in enumerate(gen_val):
                # if index == 0:
                #     continue
                self.model, images = self.model.to(device).eval(), images.to(device)
                logits = torch.sigmoid(self.model(images)).cpu().detach().numpy()
                target = [ClassifierOutputTarget(int(label)) for label in labels]
                grayscale_cams = self.cam(input_tensor=images, targets=target, aug_smooth=self.aug_smooth,
                                          eigen_smooth=self.eigen_smooth)
                hits = get_hits_per_image_label(grayscale_cams, labels, boxs, rgb_images).cpu().numpy()
                loc = get_loc_per_image_label(grayscale_cams, labels, boxs)
                target_sigmoid = [ClassifierOutputSigmoidTarget(int(label)) for label in labels]
                insert, delete = get_AUC_per_image_label_(grayscale_cams, device,
                                                          target_sigmoid, self.model, rgb_images, self.batch_size,
                                                          self.percentiles_delete, self.percentiles_insert)
                drop_increase_list = get_drop_increase_per_image(grayscale_cams, device,
                                                                 target_sigmoid, self.model, rgb_images)
                record_csv, insert, delete = record(record_csv, labels, image_ids, logits, insert, delete,
                                                    hits, loc, drop_increase_list,
                                                    self.percentiles_delete, self.percentiles_insert)

                insert_ += insert
                delete_ += delete
                hits_ += np.mean(hits)
                loc_ += np.mean(loc)
                pbar.update(1)
                pbar.set_postfix({"insert": insert_ / (index + 1),
                                  "delete": delete_ / (index + 1),
                                  "hits": hits_ / (index + 1),
                                  "loc": loc_ / (index + 1)})
                # break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("time: {}".format(total_time_str) + '\n')
        record_csv.to_csv(self.csv_save_path, index=False)


if __name__ == '__main__':
    torch.cuda.set_device(device)
    args = get_args()
    method = args.method
    # batch_size = 64
    layer = args.layer
    normalization_method = args.normalization_method
    resume = os.path.join(args.path, r"weights/coco_cam/loss_20221102123343/best_acc.pth")
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
               "layer_cam_new": Layer_CAM_NEW,
               "grad_cam_new": Grad_CAM_NEW,
               "finercam": FinerCAM,
               "kpcacam": KPCACAM,
               "mgradcam": MGradCAM,
               "polycam": PCAMpm
               # "aligngradcam": AlignGradCAM
               }
    model = get_model()
    use_cuda = True
    drop_stride = 1
    drop_num = 100 // drop_stride
    percentiles_delete = [i * drop_stride for i in range(drop_num, 0, -1)]
    percentiles_insert = [i * drop_stride for i in range(drop_num - 1, -1, -1)]
    now = datetime.datetime.now()
    time_now = now.strftime("%Y%m%d%H%M")
    save_path = os.path.join(args.savepath, method, f"{layer}{normalization_method}", time_now)
    print(save_path)

    if layer == 432:
        target_layers = [model.layer2, model.layer3, model.layer4]
    elif layer == 43:
        target_layers = [model.layer3, model.layer4]
    elif layer == 42:
        target_layers = [model.layer2, model.layer4]
    elif layer == 32:
        target_layers = [model.layer2, model.layer3]
    elif layer == 3:
        target_layers = [model.layer3]
    elif layer == 2:
        target_layers = [model.layer2]
    else:
        target_layers = [model.layer4]

    # percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cam = methods[method](model=model, target_layers=target_layers, use_cuda=use_cuda)
    cam.batch_size = 32

    if hasattr(cam, 'device'):
        cam.device = device
    else:
        # 如果CAM没有device属性，手动设置
        cam.device = device

    # 对于某些CAM方法，可能需要手动设置activations_and_gradients的设备
    if hasattr(cam, 'activations_and_grads'):
        cam.activations_and_grads.device = device

    work = start(0, 2500, os.path.join(save_path, 'record.csv'),
                 model, cam, percentiles_delete, percentiles_insert, batch_size=args.batch_size,
                 num_workers=args.num_workers,
                 val_txt=os.path.join(args.path, "weights/coco_cam/val_with_box.txt"))

    work.run()
