import argparse
import os
import datetime
import time

import cv2
import numpy as np
import torch
from thop import profile
from torch import nn
import torchvision.models
from torchvision import models
from torchvision.datasets import PCAM

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
    GradCAMElementWise, \
    FinerCAM, \
    KPCACAM, \
    PCAMpm, \
    AlignGradCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.ablation_cam_new import AblationCAMNEW
from pytorch_grad_cam.grad_cam_new import Grad_CAM_NEW
# from pytorch_grad_cam.me_cam_sub import MeCAM_SUB
# from pytorch_grad_cam.me_cam_sum import MeCAM_SUM
# from pytorch_grad_cam.me_cam_weights import MeCAM_weights
from pytorch_grad_cam.mgrad_cam import MGradCAM
from pytorch_grad_cam.score_cam_new import ScoreCAMNew
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image, show_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSigmoidTarget

# 核心：用于解决最优匹配问题
from scipy.optimize import linear_sum_assignment

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        # default="/devdata/home/homefun/DATA/voc/VOCdevkit/VOC2012/JPEGImages/2008_000041.jpg",
        # default="/devdata/home/homefun/DATA/coco_cam/train2017/000000030752.jpg",
        default="/devdata/home/homefun/DATA/coco_cam/train2017/000000170101.jpg",
        help='Input image path')
    parser.add_argument(
        '--save_path',
        type=str,
        # default="/devdata/home/homefun/pytorch_grad_cam/pic_result/",
        # default="/devdata/home/homefun/CAM-copy/pic_result_new/coco_1",
        # default="/devdata/home/homefun/CAM-copy/pic_result_new/coco_comparison/432",  # mgradcam 432
        # default="/devdata/home/homefun/CAM-copy/pic_result_new/coco_comparison/4", # mgradcam 4
        # default="/devdata/home/homefun/CAM-copy/pic_result_new/coco_compare/fullgrad/4", # fullgrad 4
        # default="/devdata/home/homefun/CAM-copy/pic_result_new/coco_compare/gradcam/4", # gradcam 4
        # default="/devdata/home/homefun/CAM-copy/pic_result_new/coco_compare/gradcam++/4", # gradcam++ 4
        # default="/devdata/home/homefun/CAM-copy/pic_result_new/coco_compare/layercam/4", # layercam 4
        # default="/devdata/home/homefun/CAM-copy/pic_result_new/coco_compare/ablationcam/4",  # ablationcam 4
        # default="/devdata/home/homefun/CAM-copy/pic_result_new/coco_compare/finercam/4", # finercam 4
        default="/devdata/home/homefun/CAM-copy/pic_result_new/coco_GT_Pred/mgradcam/432",  # mgradcam 4
        help='Save image path')
    parser.add_argument(
        '--target',
        nargs='+')
    parser.add_argument(
        '--resume',
        type=str,
        default="/devdata/home/homefun/weights/coco_cam/coco_loss_20250901193927/best_acc.pth")
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--GPU', type=int, default=2, help='device')
    parser.add_argument('--layer', type=int, default=432)
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
             'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='mgradcam',
                        choices=['gradcam', 'hirescam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad', 'finercam', 'kpcacam', 'ploycam',
                                 'aligngradcam'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam/fullgrad/finercam/kpcacam/ploycam/aligngradcam')
    # parser.add_argument('--method', type=str, default='mgradcam',
    #                     choices=['gradcam', 'hirescam', 'gradcam++',
    #                              'scorecam', 'xgradcam',
    #                              'ablationcam', 'eigencam',
    #                              'eigengradcam', 'layercam', 'fullgrad','cam_can'],
    #                     help='Can be gradcam/gradcam++/scorecam/xgradcam'
    #                          '/ablationcam/eigencam/eigengradcam/layercam/cam_can')
    # parser.add_argument('--method', type=str, default='cam_can',
    #                     choices=['gradcam', 'hirescam', 'gradcam++',
    #                              'scorecam', 'xgradcam',
    #                              'ablationcam', 'eigencam',
    #                              'eigengradcam', 'layercam', 'fullgrad','cam_can'],
    #                     help='Can be gradcam/gradcam++/scorecam/xgradcam'
    #                          '/ablationcam/eigencam/eigengradcam/layercam/cam_can')

    args = parser.parse_args()
    # args.target = [8, 10, 14, 15]
    # args.target = [1, 8, 18, 27, 35]
    args.target = [1, 47, 62, 63, 77]
    args.use_cuda = True
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        torch.cuda.set_device(args.GPU)
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

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

def load_val_list(txt_path):
    samples = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()

            img_path = parts[0]
            targets = list(map(int, parts[1].split(',')))
            H, W = int(parts[2]), int(parts[3])

            bboxes = []
            for p in parts[4:]:
                x1, x2, y1, y2, c = map(int, p.split(','))
                bboxes.append([x1, x2, y1, y2, c])

            # sample的元素组成：(image_path, targets_cls, H, W, bboxes)
            samples.append((img_path, targets, H, W, bboxes))
    return samples

# VOC / COCO 风格的固定颜色表（可自行扩展）
def get_color_by_class_id(cls_id):
    """
    为每个类别生成稳定、可区分的颜色
    cls_id: COCO 原始 ID 或 连续 ID
    return: BGR tuple
    """
    np.random.seed(cls_id)
    color = np.random.randint(0, 255, size=3)
    return int(color[0]), int(color[1]), int(color[2])

def draw_gt_bboxes(
    img,
    bboxes,
    src_w,
    src_h,
    draw_label=False  # 👈 是否显示 GT:cls
):
    """
    img: 224x224 RGB (float32, 0~1)
    bboxes: [x1, x2, y1, y2, cls] in original image scale
    """
    h, w = img.shape[:2]
    sx = w / src_w
    sy = h / src_h

    img_draw = img.copy()

    for x1, x2, y1, y2, cls in bboxes:
        px1 = int(x1 * sx)
        px2 = int(x2 * sx)
        py1 = int(y1 * sy)
        py2 = int(y2 * sy)

        color = get_color_by_class_id(cls)

        # 画框
        cv2.rectangle(
            img_draw,
            (px1, py1),
            (px2, py2),
            color,
            2
        )

        # 是否画 GT:cls
        if draw_label:
            cv2.putText(
                img_draw,
                f"GT:{cls}",
                (px1, max(py1 - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

    return img_draw

# ==========================================
# 核心算法：计算两组 CAM 的最优匹配余弦相似度
# ==========================================
def compute_optimal_cosine_similarity(cams_gt, cams_pred):
    """
    Args:
        cams_gt: List of numpy arrays (H, W), 真实标签的 CAM
        cams_pred: List of numpy arrays (H, W), 预测标签的 CAM
    Returns:
        float: 最优匹配下的平均余弦相似度
    """
    if len(cams_gt) == 0 or len(cams_pred) == 0:
        return 0.0

    # 1. 展平并归一化 (Flatten and Normalize)
    # 将每个 CAM 拉成 1D 向量，并进行 L2 归一化，这样点积即为余弦相似度
    def process_cam(cam):
        flat = cam.flatten().astype(np.float32)
        norm = np.linalg.norm(flat)
        return flat / (norm + 1e-8) # 防止除零

    vectors_gt = np.array([process_cam(c) for c in cams_gt])   # Shape: (N_gt, H*W)
    vectors_pred = np.array([process_cam(c) for c in cams_pred]) # Shape: (N_pred, H*W)

    # 2. 计算相似度矩阵 (Similarity Matrix)
    # Matrix[i, j] 表示第 i 个 GT 和第 j 个 Pred 的余弦相似度
    sim_matrix = np.dot(vectors_gt, vectors_pred.T) # Shape: (N_gt, N_pred)

    # 3. 匈牙利算法 (Hungarian Algorithm) 寻找最优匹配
    # linear_sum_assignment 寻找最小成本，所以我们将相似度取负作为成本
    cost_matrix = -sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 4. 计算平均相似度
    # 只统计匹配上的对，未匹配的（因为数量不一致）不计入分母，符合“不误导”的要求
    matched_sims = sim_matrix[row_ind, col_ind]
    avg_similarity = matched_sims.mean()

    return avg_similarity

if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"mgradcam": MGradCAM,
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
         "grad_cam_new": Grad_CAM_NEW,
         "finercam": FinerCAM,
         "kpcacam": KPCACAM,
         "ploycam": PCAMpm,
         "aligngradcam": AlignGradCAM, }

    # =======================
    # COCO ID <-> 连续 ID 映射
    # =======================
    coco_id_to_contiguous, contiguous_to_coco_id = build_coco_id_mappings()

    log_dir = os.path.join(
        args.save_path,
        str(args.layer)  # 👈 432 / 43 / 4 等
    )

    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir,
        f"cam_cost_{args.method}_layer{args.layer}.csv"
    )

    # 如果文件不存在，先写表头
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write(
                "image,method,layer,num_classes,time_ms,peak_mem_MB,flops_GFLOPs\n"
            )

    # =======================
    # 1. 构建模型（只做一次）
    # =======================
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, 80)
    model.load_state_dict(torch.load(args.resume, map_location='cpu')['model'])

    if args.use_cuda:
        model = model.cuda()
    model.eval()

    # =======================
    # 2. 选择目标层
    # =======================
    if args.layer == 1:
        target_layers = [model.layer1]
    if args.layer == 4:
        target_layers = [model.layer4]
    if args.layer == 3:
        target_layers = [model.layer3]
    if args.layer == 2:
        target_layers = [model.layer2]
    if args.layer == 43:
        target_layers = [model.layer3, model.layer4]
    if args.layer == 42:
        target_layers = [model.layer2, model.layer4]
    if args.layer == 432:
        target_layers = [model.layer2, model.layer3, model.layer4]

    # =======================
    # 3. CAM 方法
    # =======================
    cam_algorithm = methods[args.method]
    cam = cam_algorithm(
        model=model,
        target_layers=target_layers,
        use_cuda=args.use_cuda
    )
    cam.batch_size = 32

    # Finer-CAM的时候需要用到device，其它CAM类方法可以注释掉
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    if hasattr(cam, 'device'):
        cam.device = device
    else:
        # 如果CAM没有device属性，手动设置
        cam.device = device

    # 对于某些CAM方法，可能需要手动设置activations_and_gradients的设备
    if hasattr(cam, 'activations_and_grads'):
        cam.activations_and_grads.device = device

    os.makedirs(args.save_path, exist_ok=True)

    # =======================
    # 4. 遍历 val_with_box.txt
    # =======================
    val_txt = "/devdata/home/homefun/weights/coco_cam/coco_val_with_box.txt"
    # samples = load_val_list(val_txt)
    # for idx, (image_path, targets_cls) in enumerate(samples):
    samples = load_val_list(val_txt)

    # 计算余弦相似度的统计变量
    total_similarity_sum = 0.0
    valid_image_count = 0

    dummy_input = torch.randn(1, 3, 224, 224)
    if args.use_cuda:
        dummy_input = dummy_input.cuda()

    flops, params = profile(
        model,
        inputs=(dummy_input,),
        verbose=False
    )
    flops_g = flops / 1e9

    for idx, (image_path, targets_cls, H, W, bboxes) in enumerate(samples):

        print(f"[{idx+1}/{len(samples)}] Processing {image_path}, targets={targets_cls}")

        if not os.path.exists(image_path):
            print("Image not found, skip.")
            continue

        # ---- 映射 targets（COCO -> 连续）----
        mapped_targets = []
        for c in targets_cls:
            if c in coco_id_to_contiguous:
                mapped_targets.append(coco_id_to_contiguous[c])

        if len(mapped_targets) == 0:
            continue

        # ---- 映射 bboxes cls（COCO -> 连续）----
        mapped_bboxes = []
        for x1, x2, y1, y2, c in bboxes:
            if c in coco_id_to_contiguous:
                mapped_bboxes.append(
                    [x1, x2, y1, y2, coco_id_to_contiguous[c]]
                )

        # ---- 读取与预处理图像 ----
        rgb_img = cv2.imread(image_path)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255.0

        input_tensor = preprocess_image(
            rgb_img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # ---- multi-label: 显式对齐 ----
        input_tensor = input_tensor.repeat(len(mapped_targets), 1, 1, 1)

        # # ---- 多标签：复制输入 ----
        # input_tensor = torch.cat(
        #     [input_tensor for _ in mapped_targets],
        #     dim=0
        # )

        if args.use_cuda:
            input_tensor = input_tensor.cuda()

        # ①进行推理得到预测类别的CAM图
        # out = model(input_tensor)
        # out_first = out[0]
        # probs = torch.sigmoid(out_first).detach().cpu().numpy()
        #
        # # ①设置阈值法
        # threshold = 0.5 * np.max(probs)
        # pred_classes = np.where(probs >= threshold)[0]
        #
        # # ②Top-K方法
        # # topk = len(targets_cls)
        # # pred_classes = np.argsort(probs)[-topk:][::-1]
        #
        # print("Predicted classes:", pred_classes)
        # print("Their scores:", probs[pred_classes])
        # # 创建预测类别的target对象
        # pred_targets = [ClassifierOutputTarget(int(tar)) for tar in pred_classes]
        #
        # # 仅针对预测类别重新构造输入张量，因为原来的input_tensor是按真实标签展开的
        # input_tensor_pred = preprocess_image(rgb_img,
        #                                      mean=[0.485, 0.456, 0.406],
        #                                      std=[0.229, 0.224, 0.225])
        # input_tensor_pred = torch.cat([input_tensor_pred for _ in pred_classes], dim=0).cuda()
        #
        # # 计算基于预测类别的CAM
        # grayscale_cams_pred = cam(input_tensor=input_tensor_pred,
        #                           targets=pred_targets,
        #                           aug_smooth=args.aug_smooth,
        #                           eigen_smooth=args.eigen_smooth)

        # ②得到真实标签类别的CAM图
        # ---- 构建 targets ----
        targets = [ClassifierOutputTarget(t) for t in mapped_targets]

        # =====================
        # CAM 时间+空间统计
        # =====================
        if args.use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        # ---- 计算 CAM ----
        grayscale_cams = cam(
            input_tensor=input_tensor,
            targets=targets,
            aug_smooth=args.aug_smooth,
            eigen_smooth=args.eigen_smooth
        )

        # -----------------------------------------------------------
        # 计算最优匹配余弦相似度
        # -----------------------------------------------------------
        # similarity = compute_optimal_cosine_similarity(grayscale_cams, grayscale_cams_pred)
        # total_similarity_sum += similarity
        # valid_image_count += 1
        # print(f"    -> Cosine Similarity (Hungarian matched): {similarity:.4f}")

        # 生成CAM标注框并保存的代码
        if args.use_cuda:
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"[MEM] Peak GPU memory: {peak_mem:.2f} MB")
        end_time = time.time()

        cam_time = end_time - start_time
        print(f"[TIME] CAM generation: {cam_time * 1000:.2f} ms")

        # ---- 保存 CAM ----
        img_name = os.path.splitext(os.path.basename(image_path))[0]

        gt_only = draw_gt_bboxes(rgb_img, bboxes, W, H, draw_label=True)
        gt_only = cv2.cvtColor((gt_only * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        cv2.imwrite(
            os.path.join(args.save_path, f"{img_name}_GT.jpg"),
            gt_only
        )

        for i, grayscale_cam in enumerate(grayscale_cams):
            cam_image = show_cam_on_image(
                rgb_img,
                grayscale_cam,
                use_rgb=True
            )
            # 画 GT bbox
            # cam_image = draw_gt_bboxes(cam_image, bboxes, W, H)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

            save_name = f"{img_name}_{args.method}_layer{args.layer}_cls{targets_cls[i]}.jpg"
            cv2.imwrite(os.path.join(args.save_path, save_name), cam_image)

        cam_time = end_time - start_time
        time_ms = cam_time * 1000
        num_classes = len(targets_cls)

        with open(log_file, "a") as f:
            f.write(
                f"{img_name},"
                f"{args.method},"
                f"{args.layer},"
                f"{num_classes},"
                f"{time_ms:.3f},"
                f"{peak_mem:.2f},"
                f"{flops_g:.2f}\n"
            )

    # -----------------------------------------------------------
    # 输出整个数据集的平均值（余弦相似度）
    # -----------------------------------------------------------
    # if valid_image_count > 0:
    #     dataset_avg_sim = total_similarity_sum / valid_image_count
    #     print("=" * 50)
    #     print(f"Method: {args.method}")
    #     print(f"Total Images Processed: {valid_image_count}")
    #     print(f"Dataset Average Cosine Similarity: {dataset_avg_sim:.4f}")
    #     print("=" * 50)
    #
    #     # 写入日志文件
    #     with open(os.path.join(args.save_path, "similarity_report.txt"), "a") as f:
    #         f.write(
    #             f"{datetime.datetime.now()}: Method={args.method}, Layer={args.layer}, AvgSim={dataset_avg_sim:.4f}\n")
    # else:
    #     print("No valid images processed.")