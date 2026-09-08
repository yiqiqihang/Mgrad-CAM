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
import psutil
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

import cProfile
import pstats
import io
import torch.backends.cudnn as cudnn

import torchvision.transforms as T

# ============ 全局优化开关 ============
cudnn.benchmark = True  # 固定分辨率可提速
USE_CHANNELS_LAST = True

# 全局设备变量
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)


def build_coco_id_mappings():
    """
    返回:
      coco_id_to_contiguous: dict[int->int], 1..90 -> 0..79（跳过空洞）
      contiguous_to_coco_id: dict[int->int], 0..79 -> 1..90
    """

    # 建立一个映射表(1-90)---> (0-79)
    unused_ids = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83]

    coco_id_to_contiguous = {}
    contiguous_to_coco_id = {}
    contiguous_id = 0
    for coco_id in range(1, 91):
        if coco_id in unused_ids:
            continue
        coco_id_to_contiguous[coco_id] = contiguous_id
        contiguous_to_coco_id[contiguous_id] = coco_id
        contiguous_id += 1

    assert contiguous_id == 80, f"Expected 80 classes, got {contiguous_id}"
    return coco_id_to_contiguous, contiguous_to_coco_id


class CAM_Dataset(Dataset):
    def __init__(self, annotation_lines, w, h, coco_id_to_contiguous=None):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.w = w
        self.h = h
        self.coco_id_to_contiguous = coco_id_to_contiguous  # COCO ID 到连续 ID 的映射

        # ============ 数据集优化版 ============
        self.transform = T.Compose([
            T.Resize((h, w)),
            T.ToTensor(),
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        line = self.annotation_lines[index].split()
        img_path = line[0]
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(image, (self.w, self.h))
        rgb_img_float = np.float32(rgb_img) / 255

        # tensor_img = self.transform(T.ToPILImage()(rgb_img))
        """把transform放在PIL转换之外保持一致性"""
        tensor_img = self.transform(T.ToPILImage()(rgb_img))
        raw_labels = [int(i) for i in line[1].split(',')]
        if self.coco_id_to_contiguous is not None:
            label = [self.coco_id_to_contiguous[l] for l in raw_labels]
        else:
            label = raw_labels

        boxs = self.get_box(line[4:], image.shape[1], image.shape[0])
        image_id = os.path.basename(img_path.split('/')[-1])
        return tensor_img, rgb_img_float, label, image_id, boxs

    def get_box(self, boxs_str, w, h):
        boxs = np.zeros([len(boxs_str), 5])
        for index, box_str in enumerate(boxs_str):
            boxs_ = np.array([int(x) for x in box_str.split(",")])
            boxs_[0], boxs_[1] = (self.w / w) * boxs_[0], (self.w / w) * boxs_[1]
            boxs_[2], boxs_[3] = (self.h / h) * boxs_[2], (self.h / h) * boxs_[3]
            # 将 COCO 标签从 1-90 映射到 0-79
            if self.coco_id_to_contiguous is not None:
                boxs_[4] = self.coco_id_to_contiguous[boxs_[4]]
            boxs[index, :] = boxs_
        return boxs

    @staticmethod
    def collate_fn(batch):
        # 返回结构与原来一致，但减小内存拷贝
        image, rgb_img, label, image_id, boxs = list(zip(*batch))
        label_list, image_list, image_id_list, rgb_img_list, boxs_list = [], [], [], [], []
        for index, lab in enumerate(label):
            for l in lab:
                label_list.append(l)
                image_list.append(image[index])
                image_id_list.append(image_id[index])
                rgb_img_list.append(rgb_img[index][np.newaxis, :])  # 保持 numpy 小尺寸
                boxs_list.append(boxs[index][np.where(boxs[index][:, -1] == l), :][0, ..., :-1])
        return torch.stack(image_list, dim=0), np.concatenate(rgb_img_list,
                                                              axis=0), label_list, image_id_list, boxs_list


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=4, help="num_workers")  # 添加多线程工作，设置CPU核的1/2
    parser.add_argument('--profile', default=False, help='Enable profiling')  # 添加性能分析参数
    parser.add_argument('--method', type=str, default='mgradcam')
    parser.add_argument('--path', type=str, default="/devdata/home/homefun/")
    parser.add_argument('--savepath', type=str, default="/devdata/home/homefun/weights/COCO_CAM")
    parser.add_argument('--layer', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--layer', type=int, default=432)
    args = parser.parse_args()
    return args


def get_model():
    model = models.resnet50(weights=None)

    # 输出类别修改为80类
    model.fc = nn.Linear(2048, 80)

    if resume != "":
        model.load_state_dict(torch.load(resume, map_location=device)['model'])

    # 将模型移到GPU并设置为评估模式
    model = model.to(device).eval()
    if USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)

    return model


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


def boxes_to_masks(boxs_list, H, W, device):
    """
    boxs_list: list of np.ndarray [K,4] for each sample; each row = [x1,x2,y1,y2]
    return: torch.float32 masks [N,H,W] in {0,1}
    """
    masks = []
    for bset in boxs_list:
        m = torch.zeros((H, W), device=device, dtype=torch.float32)
        if bset.size > 0:
            for b in bset:
                x1, x2, y1, y2 = map(int, b[:4])
                x1 = max(0, min(W - 1, x1))
                x2 = max(0, min(W - 1, x2))
                y1 = max(0, min(H - 1, y1))
                y2 = max(0, min(H - 1, y2))
                if x2 >= x1 and y2 >= y1:
                    m[y1:y2 + 1, x1:x2 + 1] = 1.0
        masks.append(m)
    return torch.stack(masks, dim=0)


def get_hits_per_image_label(grayscale_cams, labels, boxs, rgb_images):
    # 直接在GPU上处理
    grayscale_cams = torch.from_numpy(grayscale_cams).to(device)

    # 二值化cams
    cam_list = []
    for index in range(grayscale_cams.shape[0]):
        cam = grayscale_cams[index]
        cam[cam < torch.mean(cam)] = 0
        cam[cam != 0] = 1
        cam_list.append(cam)
    cams = torch.stack(cam_list, dim=0)

    label_cam_list = []
    for index, box in enumerate(boxs):
        cam_label = torch.ones_like(cam_list[0]) * 2
        for b in box:
            cam_label[int(b[2]):int(b[3]) + 1, int(b[0]):int(b[1]) + 1] = 1
        label_cam_list.append(cam_label)

    label_cams = torch.stack(label_cam_list, dim=0)

    result = cams * label_cams
    result_true = torch.sum(
        torch.where(result == 1, torch.ones_like(result), torch.zeros_like(result)).reshape(result.shape[0], -1),
        dim=1)
    result_false = torch.sum(
        torch.where(result == 2, torch.ones_like(result), torch.zeros_like(result)).reshape(result.shape[0], -1),
        dim=1)
    result = result_true / (result_true + result_false + 1e-7)
    return result.cpu().numpy()  # 最后移回CPU


def get_loc_per_image_label(grayscale_cams, labels, boxs):
    assert grayscale_cams.shape[0] == len(labels)
    grayscale_cams = torch.from_numpy(grayscale_cams).to(device)
    label_cam_list = []
    for index, box in enumerate(boxs):
        cam_label = torch.zeros_like(grayscale_cams[0])
        for b in box:
            cam_label[int(b[2]):int(b[3]) + 1, int(b[0]):int(b[1]) + 1] = 1
        label_cam_list.append(cam_label)

    label_cams = torch.stack(label_cam_list, dim=0)

    result = grayscale_cams * label_cams

    result = [(torch.sum(re) / (torch.sum(cam) + 1e-7)).cpu().numpy() for re, cam in zip(result, grayscale_cams)]
    return result


def get_drop_data_gpu(rgb_img, cam, threshold, delete=1):
    """GPU版本的drop data生成"""
    rgb_img_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).to(device)  # [3, H, W]
    cam_tensor = torch.from_numpy(cam).to(device).repeat(3, 1, 1)  # [3, H, W]

    if delete:
        drop_data = torch.where(cam_tensor >= threshold,
                                torch.ones_like(rgb_img_tensor) * 0.5,
                                rgb_img_tensor)
    else:
        drop_data = torch.where(cam_tensor >= threshold,
                                rgb_img_tensor,
                                torch.ones_like(rgb_img_tensor) * 0.5)

    drop_data = drop_data.unsqueeze(0)  # [1, 3, H, W]
    drop_data = normalizer(drop_data)
    return drop_data


def get_drop_increase_per_image(grayscale_cams, device, targets, model, rgb_imgs):
    """GPU版本的drop increase计算"""
    rgb_imgs_tensor = torch.from_numpy(rgb_imgs).permute(0, 3, 1, 2).to(device)  # [N, 3, H, W]
    grayscale_cams_np = grayscale_cams

    threshold_list = [np.percentile(cams, 50) for cams in grayscale_cams_np]

    # 批量生成drop数据
    drop_img_list = []
    for j in range(len(rgb_imgs)):
        drop_data = get_drop_data_gpu(rgb_imgs[j], grayscale_cams_np[j], threshold_list[j], 0)
        drop_img_list.append(drop_data)

    drop_tensor = torch.cat(drop_img_list, dim=0)

    with torch.no_grad():
        # 使用GPU计算
        raw_outputs = model(normalizer(rgb_imgs_tensor)).detach()
        now_outputs = model(drop_tensor).detach()

        drop_increase_list = []
        for target, now_output, raw_output in zip(targets, now_outputs, raw_outputs):
            raw_score = target(raw_output).cpu().numpy()
            now_score = target(now_output).cpu().numpy()
            drop_increase = (raw_score - now_score) / (raw_score)
            # # 确保返回标量值
            # drop_increase = float(drop_increase)
            drop_increase_list.append(drop_increase)

    return np.array(drop_increase_list)


# def get_drop_increase_per_image(grayscale_cams, device, targets, model, rgb_imgs):
#     """GPU版本的drop increase计算"""
#     rgb_imgs_tensor = torch.from_numpy(rgb_imgs).permute(0, 3, 1, 2).to(device)  # [N, 3, H, W]
#     grayscale_cams_np = grayscale_cams
#
#     threshold_list = [np.percentile(cams, 50) for cams in grayscale_cams_np]
#
#     # 批量生成drop数据
#     drop_img_list = []
#     for j in range(len(rgb_imgs)):
#         drop_data = get_drop_data_gpu(rgb_imgs[j], grayscale_cams_np[j], threshold_list[j], 0)
#         drop_img_list.append(drop_data)
#
#     drop_tensor = torch.cat(drop_img_list, dim=0)
#
#     with torch.no_grad():
#         # 使用GPU计算
#         raw_outputs = model(normalizer(rgb_imgs_tensor)).detach()
#         now_outputs = model(drop_tensor).detach()
#
#         drop_increase_list = []
#         for target, now_output, raw_output in zip(targets, now_outputs, raw_outputs):
#             raw_score = target(raw_output.unsqueeze(0)).cpu().numpy()
#             now_score = target(now_output.unsqueeze(0)).cpu().numpy()
#             drop_increase = (raw_score - now_score) / (raw_score + 1e-7)
#             # 确保返回标量值
#             drop_increase = float(drop_increase)
#             drop_increase_list.append(drop_increase)
#
#     return np.array(drop_increase_list)

# def get_drop_increase_per_image(grayscale_cams, device, targets, model, rgb_imgs):
#     rgb_imgs = torch.from_numpy(rgb_imgs)
#     grayscale_cams = torch.from_numpy(grayscale_cams)
#     threshold_list = [np.percentile(cams, 50) for cams in grayscale_cams]
#     drop_img_list = [get_drop_data_gpu(rgb_imgs[j], grayscale_cams[j], threshold_list[j], 0) for j in range(len(rgb_imgs))]
#     drop_tensor = torch.cat(drop_img_list, dim=0)
#     with torch.no_grad():
#         raw_outputs = model(normalizer(torch.permute(rgb_imgs, [0, 3, 1, 2])).to(device)).detach().cpu()
#         now_outputs = model(drop_tensor.to(device)).detach().cpu()
#         drop_increase_list = np.array([(target(raw_output).numpy() -
#                                         target(now_output).numpy()) / target(raw_output).numpy()
#                                        for target, now_output, raw_output in zip(targets, now_outputs, raw_outputs)])
#     return drop_increase_list


def get_auc_score_gpu(model, rgb_imgs, grayscale_cams, targets, percentiles, batch_size, delete=0):
    """GPU版本的AUC分数计算"""
    threshold_list = [[np.percentile(cams, per) for per in percentiles] for cams in grayscale_cams]

    # 预分配GPU内存
    drop_batches = []
    current_batch = []
    current_count = 0

    for j in range(len(threshold_list)):
        for th in threshold_list[j]:
            drop_data = get_drop_data_gpu(rgb_imgs[j], grayscale_cams[j], th, delete)
            current_batch.append(drop_data)
            current_count += 1

            if current_count >= batch_size:
                drop_batches.append(torch.cat(current_batch, dim=0))
                current_batch = []
                current_count = 0

    if current_batch:
        drop_batches.append(torch.cat(current_batch, dim=0))

    # 批量推理
    all_outputs = []
    with torch.no_grad():
        for batch in drop_batches:
            outputs = model(batch).detach().cpu()
            all_outputs.append(outputs)

    # 重组结果
    all_outputs = torch.cat(all_outputs, dim=0)

    # 按原始结构重组
    start_idx = 0
    output_list = []
    for j in range(len(threshold_list)):
        end_idx = start_idx + len(threshold_list[j])
        output_list.append(all_outputs[start_idx:end_idx])
        start_idx = end_idx

    # 计算分数
    scores = np.array([target(output).numpy() for output, target in zip(output_list, targets)])
    return scores


def get_AUC_per_image_label_gpu(grayscale_cams, targets, model, rgb_imgs, batch_size,
                                percentiles_delete, percentiles_insert):
    """GPU版本的AUC计算"""
    scores_delete = get_auc_score_gpu(model, rgb_imgs, grayscale_cams, targets,
                                      percentiles_delete, batch_size, delete=1)
    scores_insert = get_auc_score_gpu(model, rgb_imgs, grayscale_cams, targets,
                                      percentiles_insert, batch_size, delete=0)
    return scores_insert, scores_delete


def record_fast(records, labels, image_ids, logits, insert, delete, hits, loc,
                drop_increase_list, percentiles_delete, percentiles_insert,
                contiguous_to_coco_id=None):
    insert_ = np.mean(insert, axis=1)
    delete_ = np.mean(delete, axis=1)

    for index, label in enumerate(labels):
        # # 强制降drop变成float标量
        # drop_val = float(drop_increase_list[index])
        row = {
            "image_id": image_ids[index],
            "label": label,
            "logits": logits[index, label],
            "drop": drop_increase_list[index] if drop_increase_list[index] > 0 else 0,
            "increase": 1 if drop_increase_list[index] < 0 else 0,
            # "drop": drop_val if drop_val > 0 else 0,
            # "increase": 1 if drop_val < 0 else 0,
            "insert_mean": insert_[index],
            "delete_mean": delete_[index],
            "hits": hits[index],
            "loc": loc[index]
        }

        if contiguous_to_coco_id is not None:
            row["coco_label"] = contiguous_to_coco_id[label]

        for item in range(len(percentiles_delete)):
            row[f"delete_{percentiles_delete[item]}"] = delete[index][item]
        for item in range(len(percentiles_insert)):
            row[f"insert_{percentiles_insert[item]}"] = insert[index][item]

        records.append(row)

    return records, insert_.mean(), delete_.mean()


def profile_function(func, *args, **kwargs):
    """
    分析函数性能

    参数:
        func: 要分析的函数
        *args, **kwargs: 传递给函数的参数

    返回:
        函数的执行结果
    """
    # 创建性能分析器
    profiler = cProfile.Profile()
    profiler.enable()  # 开始分析

    # 执行函数
    result = func(*args, **kwargs)

    profiler.disable()  # 停止分析

    # 创建统计对象
    s = io.StringIO()
    sortby = 'cumulative'  # 按累计时间排序
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)

    # 打印分析结果
    ps.print_stats(10)  # 打印前10个最耗时的函数
    print(s.getvalue())

    return result


class start():
    def __init__(self, start_: int, end_: int, csv_save_path: str,
                 model, cam, percentiles_delete, percentiles_insert,
                 w: int = 224,
                 h: int = 224,
                 batch_size: int = 1,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False,
                 num_workers: bool = 1,
                 # 修改成coco_dataset路径
                 val_txt: str = r"/devdata/home/homefun/weights/coco_cam/coco_val_with_box.txt",
                 coco_id_to_contiguous=None,
                 contiguous_to_coco_id=None):
        self.percentiles_insert = percentiles_insert
        self.percentiles_delete = percentiles_delete
        self.model = model.to(device)
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
        self.coco_id_to_contiguous = coco_id_to_contiguous
        self.contiguous_to_coco_id = contiguous_to_coco_id

        # 控制records写磁盘的阈值，避免内存占满
        self._flush_every = 100
        self._header_written = False

    def _safe_release_cam_hooks(self):
        # 尝试各种可能的清理函数（兼容不同版本的 pytorch-grad-cam）
        try:
            if hasattr(self.cam, 'clear_hooks'):
                self.cam.clear_hooks()
        except Exception:
            pass
        try:
            if hasattr(self.cam, 'activations_and_grads') and hasattr(self.cam.activations_and_grads, 'release'):
                self.cam.activations_and_grads.release()
        except Exception:
            pass

    def _maybe_flush_records(self, records):
        # 每 self._flush_every 条记录 flush 到磁盘以释放内存
        if len(records) >= self._flush_every:
            df = pd.DataFrame(records)
            if not self._header_written and not os.path.exists(self.csv_save_path):
                df.to_csv(self.csv_save_path, index=False, mode='w')
                self._header_written = True
            else:
                df.to_csv(self.csv_save_path, index=False, mode='a', header=False)
            records.clear()
            del df
            gc.collect()

    def run(self):
        start_time = time.time()

        print("CUDA available:", torch.cuda.is_available())
        print("Using device:", device)

        # dataloader
        with open(self.val_txt, encoding='utf-8') as f:
            val_lines = f.readlines()
        val_lines = val_lines[self.start_:self.end_ if self.end_ <= len(val_lines) else len(val_lines)]

        # 构建映射表（1-90）---> (0-80)
        val_dataset = CAM_Dataset(val_lines, self.w, self.h, self.coco_id_to_contiguous)
        # gen_val = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
        #                      persistent_workers=False, shuffle=False,
        #                      collate_fn=val_dataset.collate_fn)
        # """关键改动：persistent_workers = False, pin_memory = False减少内存膨胀"""
        gen_val = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False,
                             persistent_workers=False, shuffle=False, collate_fn=val_dataset.collate_fn)

        # # 预加载模型到GPU
        # self.model = self.model.to(device).eval()
        # if USE_CHANNELS_LAST:
        #     self.model = self.model.to(memory_format=torch.channels_last)

        # # 确保CAM的模型也在正确的设备上
        # self.cam.model = self.cam.model.to(device)
        # if hasattr(self.cam, 'device'):
        #     self.cam.device = device

        records = []
        with tqdm(total=len(gen_val)) as pbar:
            for index, (images, rgb_images, labels, image_ids, boxs) in enumerate(gen_val):
                # 异步传输到GPU
                images = images.to(device, non_blocking=True)

                # 使用混合精度加速推理
                with torch.cuda.amp.autocast():
                    logits = torch.sigmoid(self.model(images))
                logits = logits.detach().cpu().numpy()

                target = [ClassifierOutputTarget(int(label)) for label in labels]

                # 计算类激活图cam（某些 CAM 内部会注册 hook，之后要释放）
                grayscale_cams = self.cam(input_tensor=images, targets=target, aug_smooth=self.aug_smooth,
                                          eigen_smooth=self.eigen_smooth)

                # 使用GPU加速的指标计算
                hits = get_hits_per_image_label(grayscale_cams, labels, boxs, rgb_images)
                loc = get_loc_per_image_label(grayscale_cams, labels, boxs)

                target_sigmoid = [ClassifierOutputSigmoidTarget(int(label)) for label in labels]

                # 使用GPU版本的AUC计算
                insert, delete = get_AUC_per_image_label_gpu(grayscale_cams, target_sigmoid, self.model,
                                                             rgb_images, self.batch_size,
                                                             self.percentiles_delete, self.percentiles_insert)

                drop_increase_list = get_drop_increase_per_image(grayscale_cams, device,
                                                                 target_sigmoid, self.model, rgb_images)

                records, insert_avg, delete_avg = record_fast(records, labels, image_ids, logits, insert, delete,
                                                              hits, loc, drop_increase_list,
                                                              self.percentiles_delete, self.percentiles_insert,
                                                              self.contiguous_to_coco_id)

                hits_avg = np.mean(hits)
                loc_avg = np.mean(loc)
                pbar.update(1)
                pbar.set_postfix({
                    "insert": insert_avg,
                    "delete": delete_avg,
                    "hits": hits_avg,
                    "loc": loc_avg
                })

                # if index == 1 or index % 10 == 0:
                #     hits_avg = np.mean(hits)
                #     loc_avg = np.mean(loc)
                #     pbar.update(1)
                #     pbar.set_postfix({
                #         "insert": insert_avg,
                #         "delete": delete_avg,
                #         "hits": hits_avg,
                #         "loc": loc_avg
                #     })

                # # ---- 关键：清理 CAM 注册的 hook / activations 缓存 ----
                # try:
                #     self._safe_release_cam_hooks()
                # except Exception:
                #     pass

                # ---- 关键：1.定期 flush records 到磁盘，避免内存无限增长 2.当存储的内容大于4GB的时候自动存储清空----
                # self._maybe_flush_records(records)
                if psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3 > 4:
                    self._maybe_flush_records(records)

                # ---- 关键：删除临时大对象并回收内存 ----
                try:
                    del grayscale_cams
                except Exception:
                    pass
                # 删除可能占内存的大变量
                for varname in ("insert", "delete", "loc", "hits", "drop_increase_list", "logits", "rgb_images"):
                    try:
                        if varname in locals():
                            del locals()[varname]
                    except Exception:
                        pass

                gc.collect()
                # 清理 cuda 缓存（注意：这不会释放显存中正在使用的张量，只释放缓存）
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        # 循环结束后，把剩余的 records 写入磁盘
        if len(records) > 0:
            df = pd.DataFrame(records)
            if not self._header_written and not os.path.exists(self.csv_save_path):
                df.to_csv(self.csv_save_path, index=False, mode='w')
            else:
                df.to_csv(self.csv_save_path, index=False, mode='a', header=False)
            records.clear()
            del df
            gc.collect()

        self.cam.activations_and_grads.release()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("time: {}".format(total_time_str) + '\n')

        # 保存结果
        # pd.DataFrame(records).to_csv(self.csv_save_path, index=False)


if __name__ == '__main__':
    torch.cuda.set_device(device)
    args = get_args()
    method = args.method
    layer = args.layer

    # 调用训练好的coco数据集权重
    resume = os.path.join(args.path, r"weights/coco_cam/coco_loss_20250901193927/best_acc.pth")

    # 添加 COCO ID 映射
    coco_id_to_contiguous, contiguous_to_coco_id = build_coco_id_mappings()

    # methods = {"gradcam": GradCAM,
    #            "hirescam": HiResCAM,
    #            "scorecam": ScoreCAM,
    #            "scorecamnew": ScoreCAMNew,
    #            "gradcam++": GradCAMPlusPlus,
    #            "ablationcam": AblationCAM,
    #            "ablationcamnew": AblationCAMNEW,
    #            "xgradcam": XGradCAM,
    #            "eigencam": EigenCAM,
    #            "eigengradcam": EigenGradCAM,
    #            "layercam": LayerCAM,
    #            "fullgrad": FullGrad,
    #            "gradcamelementwise": GradCAMElementWise,
    #            "layer_cam_new": Layer_CAM_NEW,
    #            "grad_cam_new": Grad_CAM_NEW,
    #            "finercam": FinerCAM,
    #            "kpcacam": KPCACAM,
    #            "mgradcam": MGradCAM}
    methods = {
               "gradcam": GradCAM,
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
               "mgradcam": MGradCAM}
    # 修改model.fc层，输出类别为80类
    model = get_model()

    use_cuda = True
    drop_stride = 1
    drop_num = 100 // drop_stride
    percentiles_delete = [i * drop_stride for i in range(drop_num, 0, -1)]
    percentiles_insert = [i * drop_stride for i in range(drop_num - 1, -1, -1)]
    now = datetime.datetime.now()
    time_now = now.strftime("%Y%m%d%H%M")
    save_path = os.path.join(args.savepath, method, str(layer), time_now)
    print(save_path)
    print("CPU_K:" + str(os.cpu_count()))

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

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cam = methods[method](model=model, target_layers=target_layers, use_cuda=use_cuda)


    # 彻底确保CAM中的所有模型参数都在正确的设备上
    def ensure_model_on_device(model, device):
        """确保模型所有参数都在指定设备上"""
        model = model.to(device)
        # 递归检查所有子模块
        for module in model.modules():
            for param in module.parameters():
                if param.device != device:
                    param.data = param.data.to(device)
        return model


    # 应用设备确保
    # cam.model = ensure_model_on_device(cam.model, device)

    # 设置CAM的设备属性
    if hasattr(cam, 'device'):
        cam.device = device
    else:
        # 如果CAM没有device属性，手动设置
        cam.device = device

    # 对于某些CAM方法，可能需要手动设置activations_and_gradients的设备
    if hasattr(cam, 'activations_and_grads'):
        cam.activations_and_grads.device = device

    # cam.batch_size = 4
    # batch_size 属性保留（如果 CAM 有该属性）
    try:
        cam.batch_size = 12
    except Exception:
        pass

    # 创建工作实例
    work = start(0, 30000, os.path.join(save_path, 'record.csv'),
                 model, cam, percentiles_delete, percentiles_insert, batch_size=args.batch_size,
                 num_workers=args.num_workers,
                 val_txt=os.path.join(args.path, "weights/coco_cam/coco_val_with_box.txt"),
                 coco_id_to_contiguous=coco_id_to_contiguous,
                 contiguous_to_coco_id=contiguous_to_coco_id)

    # 在main函数末尾添加
    print(f"Using device: {device}")
    print(f"Model is on: {next(model.parameters()).device}")
    print(f"CAM device: {cam.device if hasattr(cam, 'device') else 'Not set'}")

    # 使用性能分析
    if args.profile:
        profile_function(work.run)
    else:
        work.run()
