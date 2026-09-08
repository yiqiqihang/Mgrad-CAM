# import argparse
# import math
# import tracemalloc
# from multiprocessing import Process
# import time
# import datetime
# from typing import List
#
# import cv2
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import psutil
# from torch.utils.data import Dataset, DataLoader
# import gc
# import _thread
# from torch.utils.data.dataset import T_co
# from tqdm import tqdm
# from torchvision.transforms import Normalize
# from pytorch_grad_cam import GradCAM, \
#     HiResCAM, \
#     ScoreCAM, \
#     GradCAMPlusPlus, \
#     AblationCAM, \
#     XGradCAM, \
#     EigenCAM, \
#     EigenGradCAM, \
#     LayerCAM, \
#     FullGrad, \
#     FinerCAM, \
#     KPCACAM, \
#     GradCAMElementWise
# from pytorch_grad_cam.ablation_cam_new import AblationCAMNEW
# from pytorch_grad_cam.layer_cam_new import Layer_CAM_NEW
# from pytorch_grad_cam.grad_cam_new import Grad_CAM_NEW
# from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange
# from pytorch_grad_cam.mgrad_cam import MGradCAM
# from pytorch_grad_cam.score_cam_new import ScoreCAMNew
# from pytorch_grad_cam.utils import deprocess_image
# from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget, ClassifierOutputSigmoidTarget, \
#     ClassifierOutputTarget
# from pytorch_grad_cam.metrics.road import ROADCombined, ROADLeastRelevantFirstAverage, ROADMostRelevantFirstAverage
# from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst
# import torch
# import torch.nn as nn
# import torchvision.models as models
#
# import cProfile
# import pstats
# import io
# import torch.backends.cudnn as cudnn
#
# import torchvision.transforms as T
#
# # ============ 全局优化开关 ============
# cudnn.benchmark = True  # 固定分辨率可提速
# USE_CHANNELS_LAST = True
#
# # 全局设备变量
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)
#
#
# def build_coco_id_mappings():
#     """
#     返回:
#       coco_id_to_contiguous: dict[int->int], 1..90 -> 0..79（跳过空洞）
#       contiguous_to_coco_id: dict[int->int], 0..79 -> 1..90
#     """
#     # 建立一个映射表(1-90)---> (0-79)
#     unused_ids = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83]
#
#     coco_id_to_contiguous = {}
#     contiguous_to_coco_id = {}
#     contiguous_id = 0
#     for coco_id in range(1, 91):
#         if coco_id in unused_ids:
#             continue
#         coco_id_to_contiguous[coco_id] = contiguous_id
#         contiguous_to_coco_id[contiguous_id] = coco_id
#         contiguous_id += 1
#
#     assert contiguous_id == 80, f"Expected 80 classes, got {contiguous_id}"
#     return coco_id_to_contiguous, contiguous_to_coco_id
#
#
# class CAM_Dataset(Dataset):
#     def __init__(self, annotation_lines, w, h, coco_id_to_contiguous=None):
#         self.annotation_lines = annotation_lines
#         self.length = len(annotation_lines)
#         self.w = w
#         self.h = h
#         self.coco_id_to_contiguous = coco_id_to_contiguous
#
#         # ============ 数据集优化版 ============
#         self.transform = T.Compose([
#             T.Resize((h, w)),
#             T.ToTensor(),
#         ])
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, index):
#         line = self.annotation_lines[index].split()
#         img_path = line[0]
#         image = cv2.imread(img_path)
#         if image is None:
#             image = np.zeros((self.h, self.w, 3), dtype=np.uint8)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         rgb_img = cv2.resize(image, (self.w, self.h))
#         rgb_img_float = np.float32(rgb_img) / 255
#
#         tensor_img = self.transform(T.ToPILImage()(rgb_img))
#         raw_labels = [int(i) for i in line[1].split(',')]
#         if self.coco_id_to_contiguous is not None:
#             label = [self.coco_id_to_contiguous[l] for l in raw_labels]
#         else:
#             label = raw_labels
#
#         boxs = self.get_box(line[4:], image.shape[1], image.shape[0])
#         image_id = os.path.basename(img_path.split('/')[-1])
#         return tensor_img, rgb_img_float, label, image_id, boxs
#
#     def get_box(self, boxs_str, w, h):
#         boxs = np.zeros([len(boxs_str), 5])
#         for index, box_str in enumerate(boxs_str):
#             boxs_ = np.array([int(x) for x in box_str.split(",")])
#             boxs_[0], boxs_[1] = (self.w / w) * boxs_[0], (self.w / w) * boxs_[1]
#             boxs_[2], boxs_[3] = (self.h / h) * boxs_[2], (self.h / h) * boxs_[3]
#             if self.coco_id_to_contiguous is not None:
#                 boxs_[4] = self.coco_id_to_contiguous[boxs_[4]]
#             boxs[index, :] = boxs_
#         return boxs
#
#     @staticmethod
#     def collate_fn(batch):
#         image, rgb_img, label, image_id, boxs = list(zip(*batch))
#         label_list, image_list, image_id_list, rgb_img_list, boxs_list = [], [], [], [], []
#         for index, lab in enumerate(label):
#             for l in lab:
#                 label_list.append(l)
#                 image_list.append(image[index])
#                 image_id_list.append(image_id[index])
#                 rgb_img_list.append(rgb_img[index][np.newaxis, :])
#                 boxs_list.append(boxs[index][np.where(boxs[index][:, -1] == l), :][0, ..., :-1])
#         return torch.stack(image_list, dim=0), np.concatenate(rgb_img_list,
#                                                               axis=0), label_list, image_id_list, boxs_list
#
#
# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--num_workers', type=int, default=1, help="num_workers")
#     parser.add_argument('--profile', default=False, help='Enable profiling')
#     parser.add_argument('--method', type=str, default='mgradcam')
#     parser.add_argument('--path', type=str, default="/devdata/home/homefun/")
#     parser.add_argument('--savepath', type=str, default="/devdata/home/homefun/weights/COCO_CAM_VGG")
#
#     # 【修改点】：将layer改为字符串输入，方便定义不同的融合组合，如 '543' 代表 VGG16 的 Block 5, 4, 3
#     parser.add_argument('--layer', type=str, default='5', help="VGG layers to fuse, e.g., '543', '54', '5'")
#     parser.add_argument('--batch_size', type=int, default=32)
#     args = parser.parse_args()
#     return args
#
#
# def get_model():
#     # 【修改点】：替换为 VGG16
#     print("Loading VGG16 model...")
#     model = models.vgg16(weights=None)
#
#     # VGG16的分类头(classifier)最后一层是第6层
#     in_features = model.classifier[6].in_features
#     # 输出类别修改为80类
#     model.classifier[6] = nn.Linear(in_features, 80)
#
#     if resume != "" and os.path.exists(resume):
#         try:
#             checkpoint = torch.load(resume, map_location='cpu')
#             if 'model' in checkpoint:
#                 model.load_state_dict(checkpoint['model'])
#             else:
#                 model.load_state_dict(checkpoint)
#             print(f"Successfully loaded weights from {resume}")
#         except Exception as e:
#             print(f"Warning: Failed to load weights (Ensure they are VGG16 weights). Error: {e}")
#     else:
#         print("Warning: No pre-trained weights found or path empty. Using random initialization.")
#
#     model = model.to(device).eval()
#     # if USE_CHANNELS_LAST:
#     #     model = model.to(memory_format=torch.channels_last)
#
#     return model
#
#
# # ========= 中间的各种计算函数保持你的原样优化，未做改动 =========
#
# def get_hits_per_image_label(grayscale_cams, labels, boxs, rgb_images):
#     grayscale_cams = torch.from_numpy(grayscale_cams).to(device)
#     cam_list = []
#     for index in range(grayscale_cams.shape[0]):
#         cam = grayscale_cams[index]
#         cam[cam < torch.mean(cam)] = 0
#         cam[cam != 0] = 1
#         cam_list.append(cam)
#     cams = torch.stack(cam_list, dim=0)
#
#     label_cam_list = []
#     for index, box in enumerate(boxs):
#         cam_label = torch.ones_like(cam_list[0]) * 2
#         for b in box:
#             cam_label[int(b[2]):int(b[3]) + 1, int(b[0]):int(b[1]) + 1] = 1
#         label_cam_list.append(cam_label)
#
#     label_cams = torch.stack(label_cam_list, dim=0)
#
#     result = cams * label_cams
#     result_true = torch.sum(
#         torch.where(result == 1, torch.ones_like(result), torch.zeros_like(result)).reshape(result.shape[0], -1),
#         dim=1)
#     result_false = torch.sum(
#         torch.where(result == 2, torch.ones_like(result), torch.zeros_like(result)).reshape(result.shape[0], -1),
#         dim=1)
#     result = result_true / (result_true + result_false + 1e-7)
#     return result.cpu().numpy()
#
#
# def get_loc_per_image_label(grayscale_cams, labels, boxs):
#     assert grayscale_cams.shape[0] == len(labels)
#     grayscale_cams = torch.from_numpy(grayscale_cams).to(device)
#     label_cam_list = []
#     for index, box in enumerate(boxs):
#         cam_label = torch.zeros_like(grayscale_cams[0])
#         for b in box:
#             cam_label[int(b[2]):int(b[3]) + 1, int(b[0]):int(b[1]) + 1] = 1
#         label_cam_list.append(cam_label)
#
#     label_cams = torch.stack(label_cam_list, dim=0)
#     result = grayscale_cams * label_cams
#     result = [(torch.sum(re) / (torch.sum(cam) + 1e-7)).cpu().numpy() for re, cam in zip(result, grayscale_cams)]
#     return result
#
#
# def get_drop_data_gpu(rgb_img, cam, threshold, delete=1):
#     rgb_img_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).to(device)
#     cam_tensor = torch.from_numpy(cam).to(device).repeat(3, 1, 1)
#
#     if delete:
#         drop_data = torch.where(cam_tensor >= threshold, torch.ones_like(rgb_img_tensor) * 0.5, rgb_img_tensor)
#     else:
#         drop_data = torch.where(cam_tensor >= threshold, rgb_img_tensor, torch.ones_like(rgb_img_tensor) * 0.5)
#
#     drop_data = drop_data.unsqueeze(0)
#     drop_data = normalizer(drop_data)
#     return drop_data
#
#
# def get_drop_increase_per_image(grayscale_cams, device, targets, model, rgb_imgs):
#     rgb_imgs_tensor = torch.from_numpy(rgb_imgs).permute(0, 3, 1, 2).to(device)
#     threshold_list = [np.percentile(cams, 50) for cams in grayscale_cams]
#
#     drop_img_list = []
#     for j in range(len(rgb_imgs)):
#         drop_data = get_drop_data_gpu(rgb_imgs[j], grayscale_cams[j], threshold_list[j], 0)
#         drop_img_list.append(drop_data)
#
#     drop_tensor = torch.cat(drop_img_list, dim=0)
#
#     with torch.no_grad():
#         raw_outputs = model(normalizer(rgb_imgs_tensor)).detach()
#         now_outputs = model(drop_tensor).detach()
#
#         drop_increase_list = []
#         for target, now_output, raw_output in zip(targets, now_outputs, raw_outputs):
#             raw_score = target(raw_output).cpu().numpy()
#             now_score = target(now_output).cpu().numpy()
#
#             # 【修改点】：防止分母为 0 导致 Warning 或 NAN
#             if raw_score == 0:
#                 drop_increase_list.append(0.0)
#             else:
#                 drop_increase = (raw_score - now_score) / raw_score
#                 drop_increase_list.append(drop_increase)
#
#     return np.array(drop_increase_list)
#
#
# def get_auc_score_gpu(model, rgb_imgs, grayscale_cams, targets, percentiles, batch_size, delete=0):
#     threshold_list = [[np.percentile(cams, per) for per in percentiles] for cams in grayscale_cams]
#
#     drop_batches = []
#     current_batch = []
#     current_count = 0
#
#     for j in range(len(threshold_list)):
#         for th in threshold_list[j]:
#             drop_data = get_drop_data_gpu(rgb_imgs[j], grayscale_cams[j], th, delete)
#             current_batch.append(drop_data)
#             current_count += 1
#
#             if current_count >= batch_size:
#                 drop_batches.append(torch.cat(current_batch, dim=0))
#                 current_batch = []
#                 current_count = 0
#
#     if current_batch:
#         drop_batches.append(torch.cat(current_batch, dim=0))
#
#     all_outputs = []
#     with torch.no_grad():
#         for batch in drop_batches:
#             outputs = model(batch).detach().cpu()
#             all_outputs.append(outputs)
#
#     if len(all_outputs) > 0:
#         all_outputs = torch.cat(all_outputs, dim=0)
#     else:
#         return np.zeros((len(targets), len(percentiles)))
#
#     start_idx = 0
#     output_list = []
#     for j in range(len(threshold_list)):
#         end_idx = start_idx + len(threshold_list[j])
#         output_list.append(all_outputs[start_idx:end_idx])
#         start_idx = end_idx
#
#     scores = np.array([target(output).numpy() for output, target in zip(output_list, targets)])
#     return scores
#
#
# def get_AUC_per_image_label_gpu(grayscale_cams, targets, model, rgb_imgs, batch_size,
#                                 percentiles_delete, percentiles_insert):
#     scores_delete = get_auc_score_gpu(model, rgb_imgs, grayscale_cams, targets, percentiles_delete, batch_size,
#                                       delete=1)
#     scores_insert = get_auc_score_gpu(model, rgb_imgs, grayscale_cams, targets, percentiles_insert, batch_size,
#                                       delete=0)
#     return scores_insert, scores_delete
#
#
# def record_fast(records, labels, image_ids, logits, insert, delete, hits, loc,
#                 drop_increase_list, percentiles_delete, percentiles_insert,
#                 contiguous_to_coco_id=None):
#     insert_ = np.mean(insert, axis=1)
#     delete_ = np.mean(delete, axis=1)
#
#     for index, label in enumerate(labels):
#         row = {
#             "image_id": image_ids[index],
#             "label": label,
#             "logits": logits[index, label],
#             "drop": drop_increase_list[index] if drop_increase_list[index] > 0 else 0,
#             "increase": 1 if drop_increase_list[index] < 0 else 0,
#             "insert_mean": insert_[index],
#             "delete_mean": delete_[index],
#             "hits": hits[index],
#             "loc": loc[index]
#         }
#
#         if contiguous_to_coco_id is not None:
#             row["coco_label"] = contiguous_to_coco_id[label]
#
#         for item in range(len(percentiles_delete)):
#             row[f"delete_{percentiles_delete[item]}"] = delete[index][item]
#         for item in range(len(percentiles_insert)):
#             row[f"insert_{percentiles_insert[item]}"] = insert[index][item]
#
#         records.append(row)
#
#     return records, insert_.mean(), delete_.mean()
#
#
# def profile_function(func, *args, **kwargs):
#     profiler = cProfile.Profile()
#     profiler.enable()
#     result = func(*args, **kwargs)
#     profiler.disable()
#
#     s = io.StringIO()
#     sortby = 'cumulative'
#     ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
#     ps.print_stats(10)
#     print(s.getvalue())
#
#     return result
#
#
# # ==============================================================
#
# class start():
#     def __init__(self, start_: int, end_: int, csv_save_path: str,
#                  model, cam, percentiles_delete, percentiles_insert,
#                  w: int = 224,
#                  h: int = 224,
#                  batch_size: int = 1,
#                  aug_smooth: bool = False,
#                  eigen_smooth: bool = False,
#                  num_workers: int = 1,
#                  val_txt: str = r"/devdata/home/homefun/weights/coco_cam/coco_val_with_box.txt",
#                  coco_id_to_contiguous=None,
#                  contiguous_to_coco_id=None):
#         self.percentiles_insert = percentiles_insert
#         self.percentiles_delete = percentiles_delete
#         self.model = model.to(device)
#         self.end_ = end_
#         self.start_ = start_
#         self.csv_save_path = csv_save_path
#         self.cam = cam
#         self.w = w
#         self.h = h
#         self.batch_size = batch_size
#         self.aug_smooth = aug_smooth
#         self.eigen_smooth = eigen_smooth
#         self.num_workers = num_workers
#         self.val_txt = val_txt
#         self.coco_id_to_contiguous = coco_id_to_contiguous
#         self.contiguous_to_coco_id = contiguous_to_coco_id
#         self._flush_every = 100
#         self._header_written = False
#
#     def _maybe_flush_records(self, records):
#         if len(records) >= self._flush_every:
#             df = pd.DataFrame(records)
#             if not self._header_written and not os.path.exists(self.csv_save_path):
#                 df.to_csv(self.csv_save_path, index=False, mode='w')
#                 self._header_written = True
#             else:
#                 df.to_csv(self.csv_save_path, index=False, mode='a', header=False)
#             records.clear()
#             del df
#             gc.collect()
#
#     def run(self):
#         start_time = time.time()
#         print("Using device:", device)
#
#         if not os.path.exists(self.val_txt):
#             print(f"Error: Path {self.val_txt} does not exist!")
#             return
#
#         with open(self.val_txt, encoding='utf-8') as f:
#             val_lines = f.readlines()
#         val_lines = val_lines[self.start_:self.end_ if self.end_ <= len(val_lines) else len(val_lines)]
#
#         val_dataset = CAM_Dataset(val_lines, self.w, self.h, self.coco_id_to_contiguous)
#         gen_val = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False,
#                              persistent_workers=False, shuffle=False, collate_fn=val_dataset.collate_fn)
#
#         records = []
#         with tqdm(total=len(gen_val)) as pbar:
#             for index, (images, rgb_images, labels, image_ids, boxs) in enumerate(gen_val):
#                 images = images.to(device, non_blocking=True)
#
#                 with torch.cuda.amp.autocast():
#                     logits = torch.sigmoid(self.model(images))
#                 logits = logits.detach().cpu().numpy()
#
#                 target = [ClassifierOutputTarget(int(label)) for label in labels]
#
#                 grayscale_cams = self.cam(input_tensor=images, targets=target, aug_smooth=self.aug_smooth,
#                                           eigen_smooth=self.eigen_smooth)
#
#                 hits = get_hits_per_image_label(grayscale_cams, labels, boxs, rgb_images)
#                 loc = get_loc_per_image_label(grayscale_cams, labels, boxs)
#
#                 target_sigmoid = [ClassifierOutputSigmoidTarget(int(label)) for label in labels]
#
#                 insert, delete = get_AUC_per_image_label_gpu(grayscale_cams, target_sigmoid, self.model,
#                                                              rgb_images, self.batch_size,
#                                                              self.percentiles_delete, self.percentiles_insert)
#
#                 drop_increase_list = get_drop_increase_per_image(grayscale_cams, device,
#                                                                  target_sigmoid, self.model, rgb_images)
#
#                 records, insert_avg, delete_avg = record_fast(records, labels, image_ids, logits, insert, delete,
#                                                               hits, loc, drop_increase_list,
#                                                               self.percentiles_delete, self.percentiles_insert,
#                                                               self.contiguous_to_coco_id)
#
#                 hits_avg = np.mean(hits)
#                 loc_avg = np.mean(loc)
#                 pbar.update(1)
#                 pbar.set_postfix({
#                     "insert": insert_avg,
#                     "delete": delete_avg,
#                     "hits": hits_avg,
#                     "loc": loc_avg
#                 })
#
#                 if psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3 > 4:
#                     self._maybe_flush_records(records)
#
#                 try:
#                     del grayscale_cams
#                 except Exception:
#                     pass
#                 for varname in ("insert", "delete", "loc", "hits", "drop_increase_list", "logits", "rgb_images"):
#                     try:
#                         if varname in locals():
#                             del locals()[varname]
#                     except Exception:
#                         pass
#                 gc.collect()
#
#         if len(records) > 0:
#             df = pd.DataFrame(records)
#             if not self._header_written and not os.path.exists(self.csv_save_path):
#                 df.to_csv(self.csv_save_path, index=False, mode='w')
#             else:
#                 df.to_csv(self.csv_save_path, index=False, mode='a', header=False)
#             records.clear()
#             del df
#             gc.collect()
#
#         if hasattr(self.cam, 'activations_and_grads'):
#             self.cam.activations_and_grads.release()
#
#         total_time = time.time() - start_time
#         total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#         print("time: {}".format(total_time_str) + '\n')
#
#
#
#
# if __name__ == '__main__':
#     torch.cuda.set_device(device)
#     args = get_args()
#     method = args.method
#     layer_str = str(args.layer)
#
#     # 【重要提醒】：这里务必修改为你重新训练过的 VGG16 的权重！
#     # 如果用 ResNet50 的权重去加载 VGG 会失败。如果你暂时没有VGG权重，请把这个变量置空以做测试。
#     resume = os.path.join(args.path, r"weights/coco_cam/coco_loss_vgg_20260228042618/best_acc.pth")
#
#     coco_id_to_contiguous, contiguous_to_coco_id = build_coco_id_mappings()
#
#     methods = {"gradcam": GradCAM,"hirescam": HiResCAM, "scorecam": ScoreCAM, "scorecamnew": ScoreCAMNew,
#                "gradcam++": GradCAMPlusPlus, "ablationcam": AblationCAM, "ablationcamnew": AblationCAMNEW,
#                "xgradcam": XGradCAM, "eigencam": EigenCAM, "eigengradcam": EigenGradCAM,
#                "layercam": LayerCAM, "fullgrad": FullGrad, "gradcamelementwise": GradCAMElementWise,
#                "layer_cam_new": Layer_CAM_NEW, "grad_cam_new": Grad_CAM_NEW, "finercam": FinerCAM,
#                "kpcacam": KPCACAM, "mgradcam": MGradCAM}
#
#     model = get_model()
#
#     use_cuda = True
#     drop_stride = 1
#     drop_num = 100 // drop_stride
#     percentiles_delete = [i * drop_stride for i in range(drop_num, 0, -1)]
#     percentiles_insert = [i * drop_stride for i in range(drop_num - 1, -1, -1)]
#     now = datetime.datetime.now()
#     time_now = now.strftime("%Y%m%d%H%M")
#
#     # 路径保存时加上 VGG 前缀，方便区分
#     save_path = os.path.join(args.savepath, method, f"VGG_{layer_str}", time_now)
#     print("Results will be saved to: ", save_path)
#     print("CPU_K:" + str(os.cpu_count()))
#
#     # ================= 核心修改：VGG16 层级映射系统 =================
#     # VGG16 features有30层网络。我们取每个 Block 的最后一层卷积层：
#     # Block 5 (高语义层): features
#     # Block 4 (中语义层): features[21]
#     # Block 3 (低语义层): features[14]
#     # Block 2 (更低语义): features[9]
#     vgg_layer_mapping = {
#         '5': model.features,
#         '4': model.features[23],
#         '3': model.features[16],
#         '2': model.features[9]
#     }
#
#     target_layers = []
#     # 动态解析层级输入 (比如 args.layer='543'，则循环添加 Block5, Block4, Block3)
#     for char in sorted(layer_str, reverse=True):
#         if char in vgg_layer_mapping:
#             target_layers.append(vgg_layer_mapping[char])
#             print(f"Added VGG16 Block {char} to target_layers.")
#
#     if not target_layers:
#         print("Warning: Incorrect layer argument. Defaulting to Block 5.")
#         target_layers = [model.features]
#     # ==============================================================
#
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#
#     cam = methods[method](model=model, target_layers=target_layers, use_cuda=use_cuda)
#
#     if hasattr(cam, 'device'):
#         cam.device = device
#     else:
#         cam.device = device
#
#     if hasattr(cam, 'activations_and_grads'):
#         cam.activations_and_grads.device = device
#
#     try:
#         cam.batch_size = 12
#     except Exception:
#         pass
#
#     work = start(0, 30000, os.path.join(save_path, 'record.csv'),
#                  model, cam, percentiles_delete, percentiles_insert, batch_size=args.batch_size,
#                  num_workers=args.num_workers,
#                  val_txt=os.path.join(args.path, "weights/coco_cam/coco_val_with_box.txt"),
#                  coco_id_to_contiguous=coco_id_to_contiguous,
#                  contiguous_to_coco_id=contiguous_to_coco_id)
#
#     print(f"Using device: {device}")
#     print(f"Model is on: {next(model.parameters()).device}")
#     print(f"CAM device: {cam.device if hasattr(cam, 'device') else 'Not set'}")
#
#     if args.profile:
#         profile_function(work.run)
#     else:
#         work.run()

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
import torchvision.transforms as T
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
from pytorch_grad_cam.mgrad_cam import MGradCAM
from pytorch_grad_cam.score_cam_new import ScoreCAMNew
from pytorch_grad_cam.utils.image import preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget, ClassifierOutputSigmoidTarget, \
    ClassifierOutputTarget
import torch
import torch.nn as nn
import torchvision.models as models
import torch.backends.cudnn as cudnn

# ============ 全局配置 ============
cudnn.benchmark = True
# 为了和VOC保持一致，尽量使用固定的device逻辑，这里沿用检测到的
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 定义归一化标准，VOC代码中用于preprocess_image
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalizer = Normalize(mean=mean, std=std).to(device)


def build_coco_id_mappings():
    # 建立映射表 (1-90) ---> (0-79)
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
    return coco_id_to_contiguous, contiguous_to_coco_id


class CAM_Dataset(Dataset):
    def __init__(self, annotation_lines, w, h, coco_id_to_contiguous=None):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.w = w
        self.h = h
        self.coco_id_to_contiguous = coco_id_to_contiguous

        # 【修改 1】：加入Normalize，与VOC的preprocess_image逻辑对齐
        self.transform = T.Compose([
            T.Resize((h, w)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)  # 必须加入这一步
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        line = self.annotation_lines[index].split()
        img_path = line[0]
        image = cv2.imread(img_path)
        if image is None:
            # 容错处理
            image = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # 保持与VOC一致的读取逻辑：BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 原始resize用于生成 rgb_img_float (0-1, 未归一化) 用于后续mask操作
        rgb_img = cv2.resize(image, (self.w, self.h))
        rgb_img_float = np.float32(rgb_img) / 255

        # tensor_img 用于输入模型，必须是归一化后的
        # 注意：T.ToPILImage() 接受 RGB
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
            if self.coco_id_to_contiguous is not None:
                boxs_[4] = self.coco_id_to_contiguous[boxs_[4]]
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
                # 筛选属于当前label的box
                current_box = boxs[index]
                if len(current_box) > 0:
                    matched = current_box[np.where(current_box[:, -1] == l), :][0, ..., :-1]
                else:
                    matched = np.array([])
                boxs_list.append(matched)
        return torch.stack(image_list, dim=0), np.concatenate(rgb_img_list,
                                                              axis=0), label_list, image_id_list, boxs_list


# 【修改 2】：使用 VOC 版本的 Hits 计算逻辑 (更健壮)
def get_hits_per_image_label(grayscale_cams, labels, boxs, rgb_images):
    # 确保在GPU上计算
    grayscale_cams = torch.from_numpy(grayscale_cams).to(device)
    cam_list = []

    # 1. 处理 CAM 图
    for index in range(grayscale_cams.shape[0]):
        cam = grayscale_cams[index]
        # VOC逻辑：如果CAM全黑，保持原样（虽然通常不会发生，但防止除0）
        if torch.max(cam) == 0:
            pass
        else:
            cam[cam < torch.mean(cam)] = 0
            cam[cam != 0] = 1
        cam_list.append(cam)
    cams = torch.stack(cam_list, dim=0)

    # 2. 处理 Box Ground Truth
    label_cam_list = []
    for index, box in enumerate(boxs):
        # 默认为2 (背景/Ignore)
        cam_label = torch.ones_like(grayscale_cams[0]) * 2

        # 增加对空box的检查，这在VOC代码里有
        if isinstance(box, np.ndarray) and box.size > 0:
            if box.ndim == 1:
                box = box[np.newaxis, :]
            for b in box:
                # 边界截断，防止越界
                x1, y1 = max(0, int(b[0])), max(0, int(b[2]))
                x2, y2 = min(cam_label.shape[1] - 1, int(b[1])), min(cam_label.shape[0] - 1, int(b[3]))
                cam_label[y1:y2 + 1, x1:x2 + 1] = 1

        label_cam_list.append(cam_label)

    label_cams = torch.stack(label_cam_list, dim=0)

    # 3. 计算 Hits
    result = cams * label_cams
    result_true = torch.sum(
        torch.where(result == 1, torch.ones_like(result), torch.zeros_like(result)).reshape(result.shape[0], -1),
        dim=1)
    result_false = torch.sum(
        torch.where(result == 2, torch.ones_like(result), torch.zeros_like(result)).reshape(result.shape[0], -1),
        dim=1)

    # 加上 1e-7 防止分母为0
    denominator = result_true + result_false + 1e-7
    result = result_true / denominator
    return result.cpu().numpy()


def get_loc_per_image_label(grayscale_cams, labels, boxs):
    # 保持一致
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

    label_cams = torch.stack(label_cam_list, dim=0)
    result = grayscale_cams * label_cams
    result = [(torch.sum(re) / (torch.sum(cam) + 1e-7)).cpu().numpy() for re, cam in zip(result, grayscale_cams)]
    return result


def get_drop_data_gpu(rgb_img, cam, threshold, delete=1):
    # RGB img 这里是 HWC numpy, 0-1
    rgb_img_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).to(device)
    cam_tensor = torch.from_numpy(cam).to(device).repeat(3, 1, 1)

    if delete:
        # Mask 掉的地方填 0.5
        drop_data = torch.where(cam_tensor >= threshold, torch.ones_like(rgb_img_tensor) * 0.5, rgb_img_tensor)
    else:
        drop_data = torch.where(cam_tensor >= threshold, rgb_img_tensor, torch.ones_like(rgb_img_tensor) * 0.5)

    drop_data = drop_data.unsqueeze(0)
    # 【注意】Mask完之后，再进行归一化送入模型，这步是正确的，VOC也是这么做的
    drop_data = normalizer(drop_data)
    return drop_data


def get_drop_increase_per_image(grayscale_cams, device, targets, model, rgb_imgs):
    # rgb_imgs: numpy (B, H, W, 3), range [0, 1]
    # VOC代码是: raw_input = normalizer(torch.permute(rgb_imgs, [0, 3, 1, 2]))
    # COCO原代码: raw_outputs = model(normalizer(rgb_imgs_tensor))
    # 逻辑一致，都是对原始图归一化后进模型算 raw_score

    rgb_imgs_tensor = torch.from_numpy(rgb_imgs).permute(0, 3, 1, 2).to(device)  # B, C, H, W
    threshold_list = [np.percentile(cams, 50) for cams in grayscale_cams]

    drop_img_list = []
    for j in range(len(rgb_imgs)):
        drop_data = get_drop_data_gpu(rgb_imgs[j], grayscale_cams[j], threshold_list[j], 0)
        drop_img_list.append(drop_data)

    drop_tensor = torch.cat(drop_img_list, dim=0)

    with torch.no_grad():
        # 确保输入是归一化过的
        raw_outputs = model(normalizer(rgb_imgs_tensor)).detach()
        now_outputs = model(drop_tensor).detach()

        drop_increase_list = []
        for target, now_output, raw_output in zip(targets, now_outputs, raw_outputs):
            raw_score = target(raw_output).cpu().numpy()
            now_score = target(now_output).cpu().numpy()

            if raw_score == 0:
                drop_increase_list.append(0.0)
            else:
                drop_increase = (raw_score - now_score) / raw_score
                drop_increase_list.append(drop_increase)

    return np.array(drop_increase_list)


def get_auc_score_gpu(model, rgb_imgs, grayscale_cams, targets, percentiles, batch_size, delete=0):
    # COCO原版通过batch处理显存，逻辑没问题，核心在于 get_drop_data_gpu 里已经做了 normalizer
    threshold_list = [[np.percentile(cams, per) for per in percentiles] for cams in grayscale_cams]

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

    all_outputs = []
    with torch.no_grad():
        for batch in drop_batches:
            outputs = model(batch).detach().cpu()
            all_outputs.append(outputs)

    if len(all_outputs) > 0:
        all_outputs = torch.cat(all_outputs, dim=0)
    else:
        return np.zeros((len(targets), len(percentiles)))

    start_idx = 0
    output_list = []
    for j in range(len(threshold_list)):
        end_idx = start_idx + len(threshold_list[j])
        output_list.append(all_outputs[start_idx:end_idx])
        start_idx = end_idx

    scores = np.array([target(output).numpy() for output, target in zip(output_list, targets)])
    return scores


def get_AUC_per_image_label_gpu(grayscale_cams, targets, model, rgb_imgs, batch_size,
                                percentiles_delete, percentiles_insert):
    scores_delete = get_auc_score_gpu(model, rgb_imgs, grayscale_cams, targets, percentiles_delete, batch_size,
                                      delete=1)
    scores_insert = get_auc_score_gpu(model, rgb_imgs, grayscale_cams, targets, percentiles_insert, batch_size,
                                      delete=0)
    return scores_insert, scores_delete


def record_fast(records, labels, image_ids, logits, insert, delete, hits, loc,
                drop_increase_list, percentiles_delete, percentiles_insert,
                contiguous_to_coco_id=None):
    insert_ = np.mean(insert, axis=1)
    delete_ = np.mean(delete, axis=1)

    for index, label in enumerate(labels):
        row = {
            "image_id": image_ids[index],
            "label": label,
            "logits": logits[index, label],
            "drop": drop_increase_list[index] if drop_increase_list[index] > 0 else 0,
            "increase": 1 if drop_increase_list[index] < 0 else 0,
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


class start():
    def __init__(self, start_: int, end_: int, csv_save_path: str,
                 model, cam, percentiles_delete, percentiles_insert,
                 w: int = 224, h: int = 224, batch_size: int = 1,
                 aug_smooth: bool = False, eigen_smooth: bool = False,
                 num_workers: int = 1, val_txt: str = "",
                 coco_id_to_contiguous=None, contiguous_to_coco_id=None):
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
        self._flush_every = 100
        self._header_written = False

    def _maybe_flush_records(self, records):
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
        print("Using device:", device)

        if not os.path.exists(self.val_txt):
            print(f"Error: Path {self.val_txt} does not exist!")
            return

        with open(self.val_txt, encoding='utf-8') as f:
            val_lines = f.readlines()
        val_lines = val_lines[self.start_:self.end_ if self.end_ <= len(val_lines) else len(val_lines)]

        val_dataset = CAM_Dataset(val_lines, self.w, self.h, self.coco_id_to_contiguous)
        gen_val = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                             pin_memory=False, persistent_workers=False, shuffle=False,
                             collate_fn=val_dataset.collate_fn)

        records = []
        with tqdm(total=len(gen_val)) as pbar:
            for index, (images, rgb_images, labels, image_ids, boxs) in enumerate(gen_val):
                images = images.to(device, non_blocking=True)

                # 【修改 3】：移除 autocast，使用 FP32 确保精度与 VOC 代码一致
                with torch.cuda.amp.autocast():
                    logits = torch.sigmoid(self.model(images))
                logits = torch.sigmoid(self.model(images))
                logits = logits.detach().cpu().numpy()

                target = [ClassifierOutputTarget(int(label)) for label in labels]

                grayscale_cams = self.cam(input_tensor=images, targets=target,
                                          aug_smooth=self.aug_smooth, eigen_smooth=self.eigen_smooth)

                hits = get_hits_per_image_label(grayscale_cams, labels, boxs, rgb_images)
                loc = get_loc_per_image_label(grayscale_cams, labels, boxs)

                target_sigmoid = [ClassifierOutputSigmoidTarget(int(label)) for label in labels]

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

                # 显存清理
                if psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3 > 4:
                    self._maybe_flush_records(records)

                try:
                    del grayscale_cams
                except Exception:
                    pass
                for varname in ("insert", "delete", "loc", "hits", "drop_increase_list", "logits", "rgb_images"):
                    try:
                        if varname in locals():
                            del locals()[varname]
                    except Exception:
                        pass
                gc.collect()

        if len(records) > 0:
            df = pd.DataFrame(records)
            if not self._header_written and not os.path.exists(self.csv_save_path):
                df.to_csv(self.csv_save_path, index=False, mode='w')
            else:
                df.to_csv(self.csv_save_path, index=False, mode='a', header=False)
            records.clear()
            del df
            gc.collect()

        if hasattr(self.cam, 'activations_and_grads'):
            self.cam.activations_and_grads.release()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("time: {}".format(total_time_str) + '\n')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=1, help="num_workers")
    parser.add_argument('--method', type=str, default='mgradcam')
    parser.add_argument('--path', type=str, default="/devdata/home/homefun/")
    parser.add_argument('--savepath', type=str, default="/devdata/home/homefun/weights/COCO_CAM_VGG")
    parser.add_argument('--layer', type=str, default='543', help="VGG layers e.g. '543'")
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    return args


def get_model():
    print("Loading VGG16 model...")
    model = models.vgg16(weights=None)
    in_features = model.classifier[6].in_features
    # COCO 80类
    model.classifier[6] = nn.Linear(in_features, 80)

    # 请确保这里的resume路径正确
    resume = os.path.join(args.path, r"weights/coco_cam/coco_loss_vgg_20260228042618/best_acc.pth")

    if resume != "" and os.path.exists(resume):
        try:
            checkpoint = torch.load(resume, map_location='cpu')
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Successfully loaded weights from {resume}")
        except Exception as e:
            print(f"Warning: Failed to load weights. Error: {e}")
    else:
        print("Warning: No weights found. Using random init.")

    model = model.to(device).eval()
    return model


if __name__ == '__main__':
    # 设置设备
    torch.cuda.set_device(device)
    args = get_args()
    method = args.method
    layer_str = str(args.layer)

    coco_id_to_contiguous, contiguous_to_coco_id = build_coco_id_mappings()

    methods = {"gradcam": GradCAM, "hirescam": HiResCAM, "scorecam": ScoreCAM, "scorecamnew": ScoreCAMNew,
               "gradcam++": GradCAMPlusPlus, "ablationcam": AblationCAM, "ablationcamnew": AblationCAMNEW,
               "xgradcam": XGradCAM, "eigencam": EigenCAM, "eigengradcam": EigenGradCAM,
               "layercam": LayerCAM, "fullgrad": FullGrad, "gradcamelementwise": GradCAMElementWise,
               "layer_cam_new": Layer_CAM_NEW, "grad_cam_new": Grad_CAM_NEW, "finercam": FinerCAM,
               "kpcacam": KPCACAM, "mgradcam": MGradCAM}

    model = get_model()

    use_cuda = True
    drop_stride = 1
    drop_num = 10 // drop_stride
    percentiles_delete = [i * drop_stride for i in range(drop_num, 0, -1)]
    percentiles_insert = [i * drop_stride for i in range(drop_num - 1, -1, -1)]
    now = datetime.datetime.now()
    time_now = now.strftime("%Y%m%d%H%M")

    save_path = os.path.join(args.savepath, method, f"VGG_{layer_str}", time_now)
    print("Results will be saved to: ", save_path)

    # 层级映射
    vgg_layer_mapping = {
        '5': model.features,
        '4': model.features[23],
        '3': model.features[16],
        '2': model.features[9]
    }

    target_layers = []
    for char in sorted(layer_str, reverse=True):
        if char in vgg_layer_mapping:
            target_layers.append(vgg_layer_mapping[char])
            print(f"Added VGG16 Block {char}")

    if not target_layers:
        target_layers = [model.features]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cam = methods[method](model=model, target_layers=target_layers, use_cuda=use_cuda)
    if hasattr(cam, 'device'):
        cam.device = device
    else:
        # 如果CAM没有device属性，手动设置
        cam.device = device

    # 对于某些CAM方法，可能需要手动设置activations_and_gradients的设备
    if hasattr(cam, 'activations_and_grads'):
        cam.activations_and_grads.device = device
    try:
        cam.batch_size = 32
    except:
        pass

    work = start(0, 30000, os.path.join(save_path, 'record.csv'),
                 model, cam, percentiles_delete, percentiles_insert, batch_size=args.batch_size,
                 num_workers=args.num_workers,
                 val_txt=os.path.join(args.path, "weights/coco_cam/coco_val_with_box.txt"),
                 coco_id_to_contiguous=coco_id_to_contiguous,
                 contiguous_to_coco_id=contiguous_to_coco_id)

    work.run()