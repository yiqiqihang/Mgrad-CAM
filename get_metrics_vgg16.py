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


# 统一设备设置，优先使用 cuda:2，否则尝试 cuda:1，最后 cpu
if torch.cuda.is_available():
    if torch.cuda.device_count() > 2:
        device_str = 'cuda:2'
    else:
        device_str = 'cuda:0'
else:
    device_str = 'cpu'
device = torch.device(device_str)

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
        # 增加容错，防止图片读取失败
        if image_raw is None:
            # 如果读取失败，返回一个全黑图像防止报错，或者你可以选择 raise error
            print(f"Warning: Failed to load image {line[0]}")
            image_raw = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        w_raw, h_raw = image_raw.shape[1], image_raw.shape[0]
        image_raw = cv2.resize(image_raw, [self.w, self.h])
        image_raw = image_raw[:, :, ::-1]  # BGR to RGB
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
                # 处理box为空的情况
                current_box = boxs[index]
                if len(current_box) > 0:
                    matched_box = current_box[np.where(current_box[:, -1] == l), :][0, ..., :-1]
                else:
                    matched_box = np.array([])
                boxs_list.append(matched_box)
        return torch.stack(image_list, dim=0), np.concatenate(rgb_img_list,
                                                              axis=0), label_list, image_id_list, boxs_list


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=1, help="num_workers")
    parser.add_argument('--method', type=str, default='mgradcam')
    parser.add_argument('--path', type=str, default="/devdata/home/homefun/")
    parser.add_argument('--savepath', type=str, default="/devdata/home/homefun/weights/CAM_VGG")
    # 修改说明：建议layer参数传入字符串，例如 '543' 代表融合 VGG 的 Block 5, 4, 3
    parser.add_argument('--layer', type=str, default='543', help="Layers to fuse, e.g. 543 for Block 5,4,3")
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    return args


def get_model(resume_path=""):
    # 修改为 VGG16
    print("Loading VGG16 model...")
    model = models.vgg16(weights=None)

    # VGG16 的分类器最后以一层 Linear(4096, num_classes) 结束
    # 原代码中 dim=2048 是 ResNet 的，VGG16 fc6/fc7 输出是 4096
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 20)

    if resume_path != "" and os.path.exists(resume_path):
        print(f"Loading weights from {resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location='cpu')
            # 兼容有些权重保存时包含 'model' key，有些直接是 state_dict
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading weights: {e}. Please ensure the weights match VGG16 architecture.")
    else:
        print("Warning: No weights loaded or path does not exist. Using random init.")

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
    grayscale_cams = torch.from_numpy(grayscale_cams).to(device)

    cam_list = []
    for index in range(grayscale_cams.shape[0]):
        cam = grayscale_cams[index]
        # 避免全0导致的错误
        if torch.max(cam) == 0:
            cam_list.append(cam)
            continue

        cam[cam < torch.mean(cam)] = 0
        cam[cam != 0] = 1
        cam_list.append(cam)
    cams = torch.stack(cam_list, dim=0).to(device)

    label_cam_list = []
    for index, box in enumerate(boxs):
        cam_label = torch.ones_like(cams[0]) * 2  # 默认为2 (背景/无关)

        # 检查box是否为空
        if isinstance(box, np.ndarray) and box.size > 0:
            if box.ndim == 1:
                box = box[np.newaxis, :]
            for b in box:
                # 边界检查
                x1, y1 = max(0, int(b[0])), max(0, int(b[2]))
                x2, y2 = min(cam_label.shape[1] - 1, int(b[1])), min(cam_label.shape[0] - 1, int(b[3]))
                cam_label[y1:y2 + 1, x1:x2 + 1] = 1

        label_cam_list.append(cam_label)

    label_cams = torch.stack(label_cam_list, dim=0).to(device)

    result = cams * label_cams
    result_true = torch.sum(
        torch.where(result == 1, torch.ones_like(result), torch.zeros_like(result)).reshape(result.shape[0], -1),
        dim=1)
    result_false = torch.sum(
        torch.where(result == 2, torch.ones_like(result), torch.zeros_like(result)).reshape(result.shape[0], -1),
        dim=1)

    # 防止除零
    denominator = result_true + result_false + 1e-7
    result = result_true / denominator
    return result


def get_loc_per_image_label(grayscale_cams, labels, boxs):
    assert grayscale_cams.shape[0] == len(labels)
    grayscale_cams = torch.from_numpy(grayscale_cams).to(device)
    label_cam_list = []
    for index, box in enumerate(boxs):
        cam_label = torch.zeros_like(grayscale_cams[0])
        if isinstance(box, np.ndarray) and box.size > 0:
            if box.ndim == 1:
                box = box[np.newaxis, :]
            for b in box:
                x1, y1 = max(0, int(b[0])), max(0, int(b[2]))
                x2, y2 = min(cam_label.shape[1] - 1, int(b[1])), min(cam_label.shape[0] - 1, int(b[3]))
                cam_label[y1:y2 + 1, x1:x2 + 1] = 1

        label_cam_list.append(cam_label)

    label_cams = torch.stack(label_cam_list, dim=0).to(device)

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
        # Permute input for model: (B, H, W, C) -> (B, C, H, W)
        raw_input = normalizer(torch.permute(rgb_imgs, [0, 3, 1, 2])).to(device)
        raw_outputs = model(raw_input).detach().cpu()

        now_outputs = model(drop_tensor.to(device)).detach().cpu()

        drop_increase_list = []
        for target, now_output, raw_output in zip(targets, now_outputs, raw_outputs):
            raw_score = target(raw_output).numpy()
            now_score = target(now_output).numpy()
            # 避免 raw_score 为 0
            if raw_score == 0:
                drop_increase_list.append(0)
            else:
                drop_increase_list.append((raw_score - now_score) / raw_score)

        drop_increase_list = np.array(drop_increase_list)
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

    drop_data = torch.unsqueeze(drop_data, dim=0)
    drop_data = normalizer(drop_data)
    return drop_data


def get_auc_score(model, device, rgb_imgs, grayscale_cams, targets, percentiles, batch_size, delete=0):
    threshold_list = [[np.percentile(cams, per) for per in percentiles] for cams in grayscale_cams]
    drop_img_list = [[get_drop_data(rgb_imgs[j],
                                    grayscale_cams[j], th, delete=delete) for th in
                      threshold_list[j]] for j in range(len(threshold_list))]
    # 展平列表
    flat_drop_imgs = [item for sublist in drop_img_list for item in sublist]
    drop_tensor = torch.cat(flat_drop_imgs, dim=0)

    output_list = []
    with torch.no_grad():
        for i in range(math.ceil(drop_tensor.shape[0] / batch_size)):
            end_idx = min(batch_size * (i + 1), drop_tensor.shape[0])
            x = drop_tensor[batch_size * i: end_idx, ...]
            x = x.to(device)
            output_list.append(model(x).detach().cpu())

        if len(output_list) > 0:
            output_list = torch.cat(output_list, dim=0)

            # 重组 output 对应到每张图片
            scores = []
            current_idx = 0
            for j in range(len(drop_img_list)):
                num_variations = len(drop_img_list[j])
                outputs_per_img = output_list[current_idx: current_idx + num_variations]
                current_idx += num_variations

                # 计算该图片的target score
                score_row = [targets[j](out).numpy() for out in outputs_per_img]
                scores.append(score_row)
            scores = np.array(scores)
        else:
            scores = np.zeros((len(rgb_imgs), len(percentiles)))

    return scores


def get_AUC_per_image_label_(grayscale_cams, device, targets, model, rgb_imgs, batch_size,
                             percentiles_delete, percentiles_insert):
    rgb_imgs = torch.from_numpy(rgb_imgs)
    grayscale_cams = torch.from_numpy(grayscale_cams)
    scores_delete = get_auc_score(model, device, rgb_imgs, grayscale_cams, targets,
                                  percentiles_delete, batch_size, delete=1)
    scores_insert = get_auc_score(model, device, rgb_imgs, grayscale_cams, targets,
                                  percentiles_insert, batch_size, delete=0)
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
                 num_workers: int = 1,
                 val_txt: str = ""):
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

        # dataloader
        if not os.path.exists(self.val_txt):
            print(f"Error: Validation file not found at {self.val_txt}")
            return

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

        print(f"Starting processing on device: {device}")

        with tqdm(total=len(gen_val)) as pbar:
            for index, (images, rgb_images, labels, image_ids, boxs) in enumerate(gen_val):
                self.model, images = self.model.to(device).eval(), images.to(device)
                logits = torch.sigmoid(self.model(images)).cpu().detach().numpy()
                target = [ClassifierOutputTarget(int(label)) for label in labels]

                # 计算 CAM
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

                record_csv, insert_batch, delete_batch = record(record_csv, labels, image_ids, logits, insert, delete,
                                                                hits, loc, drop_increase_list,
                                                                self.percentiles_delete, self.percentiles_insert)

                insert_ += insert_batch
                delete_ += delete_batch
                hits_ += np.mean(hits)
                loc_ += np.mean(loc)
                pbar.update(1)
                pbar.set_postfix({"insert": insert_ / (index + 1),
                                  "delete": delete_ / (index + 1),
                                  "hits": hits_ / (index + 1),
                                  "loc": loc_ / (index + 1)})

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("time: {}".format(total_time_str) + '\n')
        record_csv.to_csv(self.csv_save_path, index=False)


if __name__ == '__main__':
    # 确保在主程序开始前设置好设备
    torch.cuda.set_device(device)
    args = get_args()
    method = args.method

    # 将 layer 转换为字符串处理，方便组合 '543' 这种形式
    layer_str = str(args.layer)

    # 这里的 resume 路径必须是 VGG16 的权重路径！
    # 请务必修改为您训练好的 VGG16 权重路径
    resume = os.path.join(args.path, r"weights/coco_cam/loss_vgg_20260225184641/best_acc.pth")

    # 如果没有 VGG 权重，建议先把 resume 设为空字符串以跑通流程测试
    # resume = ""

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
               }

    model = get_model(resume)
    use_cuda = True if torch.cuda.is_available() else False

    drop_stride = 1
    drop_num = 100 // drop_stride
    percentiles_delete = [i * drop_stride for i in range(drop_num, 0, -1)]
    percentiles_insert = [i * drop_stride for i in range(drop_num - 1, -1, -1)]

    now = datetime.datetime.now()
    time_now = now.strftime("%Y%m%d%H%M")
    save_path = os.path.join(args.savepath, method, f"VGG_{layer_str}", time_now)
    print(f"Results will be saved to: {save_path}")

    # ================= 核心修改：VGG16 层级映射 =================
    # VGG16 features 有 31 层 (0-30)。
    # Block 5 (High Semantics): features[28] 是最后一个卷积层 (before ReLU & MaxPool)
    # Block 4 (Mid Semantics): features[21] 是 Block4 的最后一个卷积层
    # Block 3 (Low/Mid Semantics): features[14] 是 Block3 的最后一个卷积层



    target_layers = []
    # 根据 args.layer 字符串（如 '543'）来动态添加目标层

    if '5' in layer_str:
        # target_layers.append(model.features[28])
        target_layers.append(model.features)
        print("Added Block 5 (High Level)")
    if '4' in layer_str:
        target_layers.append(model.features[23])
        print("Added Block 4 (Mid Level)")
    if '3' in layer_str:
        target_layers.append(model.features[16])
        print("Added Block 3 (Low Level)")
    if '2' in layer_str:  # 如果想试更底层的
        target_layers.append(model.features[9])
        print("Added Block 2 (Low Level)")

    if not target_layers:
        print("Error: No valid layers selected. Defaulting to Block 5.")
        target_layers = [model.features[28]]
    # ==========================================================

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

    val_txt_path = os.path.join(args.path, "weights/coco_cam/val_with_box.txt")

    work = start(0, 2500, os.path.join(save_path, 'record.csv'),
                 model, cam, percentiles_delete, percentiles_insert,
                 batch_size=args.batch_size, num_workers=args.num_workers,
                 val_txt=val_txt_path)

    work.run()