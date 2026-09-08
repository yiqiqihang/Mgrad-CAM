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

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
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
        # 注意：这里可能会有读取失败的情况，建议加个判定，不过为了保持和你原代码一致，暂不修改
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
    parser.add_argument('--num_workers', type=int, default=4, help="num_workers")  # 建议稍微调高
    parser.add_argument('--method', type=str, default='mgradcam')
    parser.add_argument('--path', type=str, default="/devdata/home/homefun/")
    parser.add_argument('--savepath', type=str, default="/devdata/home/homefun/weights/CAM")
    parser.add_argument('--layer', type=int, default=432)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    return args


def get_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, 20)
    # 请确保此路径正确，或者传入参数
    resume = os.path.join(args.path, r"weights/coco_cam/loss_20221102123343/best_acc.pth")
    if os.path.exists(resume):
        model.load_state_dict(torch.load(resume, map_location='cpu')['model'])
    else:
        print(f"Warning: Model weights not found at {resume}")
    return model.eval()


def get_hits_per_image_label(grayscale_cams, labels, boxs, rgb_images):
    # 修改：为了防止boxs是空数组导致的shape错误，增加健壮性
    assert grayscale_cams.shape[0] == len(labels)
    grayscale_cams = torch.from_numpy(grayscale_cams).cuda()

    cam_list = []
    for index in range(grayscale_cams.shape[0]):
        cam = grayscale_cams[index]
        # 简单的二值化
        threshold = torch.mean(cam)
        cam = torch.where(cam < threshold, torch.tensor(0.0).cuda(), cam)
        cam = torch.where(cam != 0, torch.tensor(1.0).cuda(), cam)
        cam_list.append(cam)
    cams = torch.stack(cam_list, dim=0).cuda()

    label_cam_list = []
    for index, box in enumerate(boxs):
        cam_label = torch.ones_like(cam_list[0]) * 2
        # 检查box是否为空 (预测错误时传入空box)
        if len(box) > 0:
            for b in box:
                # 边界保护
                x1, y1 = int(max(0, b[0])), int(max(0, b[2]))
                x2, y2 = int(min(cam_label.shape[1] - 1, b[1])), int(min(cam_label.shape[0] - 1, b[3]))
                cam_label[y1:y2 + 1, x1:x2 + 1] = 1

        label_cam_list.append(cam_label)

    label_cams = torch.stack(label_cam_list, dim=0).cuda()

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
    grayscale_cams = torch.from_numpy(grayscale_cams).cuda()
    label_cam_list = []
    for index, box in enumerate(boxs):
        cam_label = torch.zeros_like(grayscale_cams[0])
        if len(box) > 0:
            for b in box:
                x1, y1 = int(max(0, b[0])), int(max(0, b[2]))
                x2, y2 = int(min(cam_label.shape[1] - 1, b[1])), int(min(cam_label.shape[0] - 1, b[3]))
                cam_label[y1:y2 + 1, x1:x2 + 1] = 1

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

        # 修正：当模型输出非常小时，防止除以接近0的数
        drop_increase_list = []
        for target, now_output, raw_output in zip(targets, now_outputs, raw_outputs):
            raw_score = target(raw_output).numpy()
            now_score = target(now_output).numpy()

            # 简单的防止除零
            if abs(raw_score) < 1e-7:
                raw_score = 1e-7

            drop_increase_list.append((raw_score - now_score) / raw_score)

    return np.array(drop_increase_list)


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

    # 扁平化所有需要推理的masked images
    # 结构: [Batch x Percentiles]
    flat_drop_list = []
    for img_group in drop_img_list:
        flat_drop_list.extend(img_group)  # 添加每个percentile的tensor

    if len(flat_drop_list) == 0:
        return np.array([])

    drop_tensor = torch.cat(flat_drop_list, dim=0)

    output_list = []
    with torch.no_grad():
        # 按小批次推理，防止显存爆炸
        for i in range(math.ceil(drop_tensor.shape[0] / batch_size)):
            x = drop_tensor[batch_size * i: batch_size * (i + 1)].to(device)
            output_list.append(model(x).detach().cpu())

    all_outputs = torch.cat(output_list, dim=0)

    # 将输出重组回 [Batch, Percentiles]
    scores = []
    num_percentiles = len(percentiles)

    for i in range(len(rgb_imgs)):  # 对于原始batch中的每张图
        target = targets[i]
        start_idx = i * num_percentiles
        end_idx = (i + 1) * num_percentiles

        img_outputs = all_outputs[start_idx:end_idx]
        img_scores = [target(out).numpy() for out in img_outputs]
        scores.append(img_scores)

    return np.array(scores)


def get_AUC_per_image_label_(grayscale_cams, device, targets, model, rgb_imgs, batch_size,
                             percentiles_delete, percentiles_insert):
    rgb_imgs = torch.from_numpy(rgb_imgs)
    grayscale_cams = torch.from_numpy(grayscale_cams)

    # 注意：这里调用时不要把device传错位置
    scores_delete = get_auc_score(model, device, rgb_imgs, grayscale_cams, targets,
                                  percentiles_delete, batch_size, delete=1)
    scores_insert = get_auc_score(model, device, rgb_imgs, grayscale_cams, targets,
                                  percentiles_insert, batch_size, delete=0)
    return scores_insert, scores_delete


def record(record_csv, labels, image_ids, logits, insert, delete, hits, loc, drop_increase_list,
           percentiles_delete, percentiles_insert, original_indices_in_batch):
    insert_mean = np.mean(insert, axis=1)
    delete_mean = np.mean(delete, axis=1)

    for i in range(len(labels)):
        # original_idx 用于获取该预测对应的原始logits
        orig_idx = original_indices_in_batch[i]
        label = labels[i]  # 这里的label实际上是 pred_class

        new_column = pd.Series(dtype='float64')
        new_column["image_id"] = image_ids[i]
        new_column["label"] = label  # 记录预测的label
        new_column["logits"] = logits[orig_idx, label]  # 获取原始logits中对应类别的分数
        new_column["drop"] = drop_increase_list[i] if drop_increase_list[i] > 0 else 0
        new_column["increase"] = 1 if drop_increase_list[i] < 0 else 0

        for item in range(len(percentiles_delete)):
            new_column["delete_{}".format(percentiles_delete[item])] = delete[i][item]
        for item in range(len(percentiles_insert)):
            new_column["insert_{}".format(percentiles_insert[item])] = insert[i][item]

        new_column["insert_mean"] = insert_mean[i]
        new_column["delete_mean"] = delete_mean[i]
        new_column["hits"] = hits[i]
        new_column["loc"] = loc[i]

        record_csv = pd.concat([record_csv, pd.DataFrame(new_column).T], ignore_index=True)

    return record_csv, np.sum(insert_mean) / len(labels), np.sum(delete_mean) / len(labels)


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
        # 建议使用传入的device，而不是硬编码
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(self.val_txt, encoding='utf-8') as f:
            val_lines = f.readlines()
        val_lines = val_lines[self.start_:self.end_ if self.end_ <= len(val_lines) else len(val_lines)]
        val_dataset = CAM_Dataset(val_lines, self.w, self.h)
        gen_val = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                             collate_fn=val_dataset.collate_fn)

        record_csv = pd.DataFrame()
        insert_ = 0
        delete_ = 0
        hits_ = 0
        loc_ = 0

        count = 0

        with tqdm(total=len(gen_val)) as pbar:
            for index, (images, rgb_images, labels, image_ids, boxs) in enumerate(gen_val):

                # 1. 基础推理
                self.model, images = self.model.to(device).eval(), images.to(device)
                logits = torch.sigmoid(self.model(images)).cpu().detach().numpy()

                # --- 核心修改：构建基于预测类别的输入 ---

                pred_classes_input = []  # 存储预测的类别ID
                images_input_list = []  # 存储对应的图像Tensor
                rgb_imgs_input_list = []  # 存储对应的RGB numpy array
                boxs_input_list = []  # 存储对应的GT Box (如果匹配的话)
                original_indices = []  # 记录属于batch里的哪张图
                image_ids_input = []  # 记录Image ID字符串

                # 遍历当前Batch中的每一张图
                for b_idx in range(images.shape[0]):

                    # 策略：选择 Top-1 预测类别 (Unsupervised explanation standard)
                    # 这样可以直接回应Reviewer：我们使用了模型预测置信度最高的类别
                    pred_label = np.argmax(logits[b_idx])

                    # 添加到输入列表
                    pred_classes_input.append(pred_label)
                    images_input_list.append(images[b_idx])
                    rgb_imgs_input_list.append(rgb_images[b_idx])
                    original_indices.append(b_idx)
                    image_ids_input.append(image_ids[b_idx])

                    # 处理 Box 用于计算 Hits/Loc
                    # 逻辑：如果 预测类别 == 当前样本的GT标签，则使用GT box。
                    # 如果不相等，说明预测错了（或者该dataloader只包含另一类别的box），此时没有GT box用于定位评估。
                    # 在这种情况下，传入空box，Loc指标应为0。
                    if pred_label == labels[b_idx]:
                        boxs_input_list.append(boxs[b_idx])
                    else:
                        boxs_input_list.append(np.array([]))  # 空数组

                # 将列表转为 Tensor/Arrays
                if len(images_input_list) == 0:
                    pbar.update(1)
                    continue

                images_input_tensor = torch.stack(images_input_list, dim=0)
                rgb_imgs_input_arr = np.array(rgb_imgs_input_list)

                targets_obj = [ClassifierOutputTarget(int(cls)) for cls in pred_classes_input]
                targets_sigmoid = [ClassifierOutputSigmoidTarget(int(cls)) for cls in pred_classes_input]

                # 2. 生成 CAM (针对预测类别)
                # 此时 images_input_tensor 和 targets_obj是一一对应的
                grayscale_cams = self.cam(
                    input_tensor=images_input_tensor,
                    targets=targets_obj,
                    aug_smooth=self.aug_smooth,
                    eigen_smooth=self.eigen_smooth
                )

                # 3. 计算指标
                # 请注意参数已经是重组后的 input list
                hits = get_hits_per_image_label(grayscale_cams, pred_classes_input, boxs_input_list,
                                                rgb_imgs_input_arr).cpu().numpy()
                loc = get_loc_per_image_label(grayscale_cams, pred_classes_input, boxs_input_list)

                # 计算 insert / delete (Faithfulness)
                insert, delete = get_AUC_per_image_label_(
                    grayscale_cams, device, targets_sigmoid, self.model,
                    rgb_imgs_input_arr, self.batch_size,
                    self.percentiles_delete, self.percentiles_insert
                )

                # 计算 drop / increase
                drop_increase_list = get_drop_increase_per_image(
                    grayscale_cams, device, targets_sigmoid,
                    self.model, rgb_imgs_input_arr
                )

                # 4. 记录结果
                # 需要传入 original_indices 以便在 record函数里查回原始 logits
                record_csv, batch_insert, batch_delete = record(
                    record_csv, pred_classes_input, image_ids_input, logits,
                    insert, delete, hits, loc, drop_increase_list,
                    self.percentiles_delete, self.percentiles_insert,
                    original_indices
                )

                count += 1
                insert_ += batch_insert
                delete_ += batch_delete
                hits_ += np.mean(hits)
                loc_ += np.mean(loc)

                pbar.update(1)
                pbar.set_postfix({
                    "ins": insert_ / count,
                    "del": delete_ / count,
                    "hit": hits_ / count,
                    "loc": loc_ / count
                })

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("time: {}".format(total_time_str) + '\n')

        # 保证目录存在
        os.makedirs(os.path.dirname(self.csv_save_path), exist_ok=True)
        record_csv.to_csv(self.csv_save_path, index=False)


if __name__ == '__main__':
    # 确保主程序入口也设置一下
    torch.cuda.set_device(device)  # device defined globally above

    args = get_args()
    method = args.method
    layer = args.layer

    # 路径检查
    path_root = args.path
    # resume = os.path.join(path_root, r"weights/coco_cam/loss_20221102123343/best_acc.pth")

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
    model = get_model()
    use_cuda = True

    drop_stride = 5  # 建议步长稍微大一点，节省时间，比如5或10
    drop_num = 100 // drop_stride
    percentiles_delete = [i * drop_stride for i in range(drop_num, 0, -1)]
    percentiles_insert = [i * drop_stride for i in range(drop_num - 1, -1, -1)]

    now = datetime.datetime.now()
    time_now = now.strftime("%Y%m%d%H%M")
    save_path = os.path.join(args.savepath, method, str(layer), time_now)
    print(f"Save Path: {save_path}")

    if layer == 432:
        target_layers = [model.layer2, model.layer3, model.layer4]
    elif layer == 43:
        target_layers = [model.layer3, model.layer4]
    elif layer == 42:
        target_layers = [model.layer2, model.layer4]
    else:
        target_layers = [model.layer4]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    try:
        cam_constructor = methods[method]
        cam = cam_constructor(model=model, target_layers=target_layers, use_cuda=use_cuda)

        # 兼容性设置
        cam.batch_size = 32
        if hasattr(cam, 'device'):
            cam.device = device
        if hasattr(cam, 'activations_and_grads'):
            cam.activations_and_grads.device = device

        csv_full_path = os.path.join(save_path, 'record_pred.csv')
        val_txt_path = os.path.join(args.path, "weights/coco_cam/val_with_box.txt")

        work = start(0, 2500, csv_full_path,
                     model, cam, percentiles_delete, percentiles_insert,
                     batch_size=args.batch_size, num_workers=args.num_workers,
                     val_txt=val_txt_path)

        work.run()

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()