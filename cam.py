import argparse
import os
import datetime
import time
import torch.nn.functional as F

import cv2
import numpy as np
import torch
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        # default="/devdata/home/homefun/DATA/voc/VOCdevkit/VOC2012/JPEGImages/2008_000041.jpg",
        # default="/devdata/home/homefun/DATA/voc/VOCdevkit/VOC2012/JPEGImages/2008_003136.jpg",
        # default="/devdata/home/homefun/DATA/voc/VOCdevkit/VOC2012/JPEGImages/2008_000834.jpg",
        default="/devdata/home/homefun/DATA/voc/VOCdevkit/VOC2012/JPEGImages/2008_007138.jpg",
        help='Input image path')
    parser.add_argument(
        '--save_path',
        type=str,
        # default="/devdata/home/homefun/pytorch_grad_cam/pic_result/",
        default="/devdata/home/homefun/CAM-copy/pic_result/",
        help='Save image path')
    parser.add_argument(
        '--target',
        nargs='+')
    parser.add_argument(
        '--resume',
        type=str,
        default="/devdata/home/homefun/weights/coco_cam/loss_20221102123343/best_acc.pth")
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--GPU', type=int, default=0, help='device')
    parser.add_argument('--layer', type=int, default=4)
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
    # args.target = [10,14,8]
    # args.target = [14, 3]
    args.target = [13, 14, 6]
    args.use_cuda = True
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        torch.cuda.set_device(args.GPU)
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    # methods = \
    #     {"gradcam": GradCAM,
    #      "hirescam": HiResCAM,
    #      "scorecam": ScoreCAM,
    #      "scorecamnew": ScoreCAMNew,
    #      "gradcam++": GradCAMPlusPlus,
    #      "ablationcam": AblationCAM,
    #      "ablationcamnew": AblationCAMNEW,
    #      "xgradcam": XGradCAM,
    #      "eigencam": EigenCAM,
    #      "eigengradcam": EigenGradCAM,
    #      "layercam": LayerCAM,
    #      "fullgrad": FullGrad,
    #      "gradcamelementwise": GradCAMElementWise,
    #      "grad_cam_new": Grad_CAM_NEW,
    #      "me_cam_w": MeCAM_weights,
    #      "me_cam_sum":MeCAM_SUM,
    #      "me_cam_sub":MeCAM_SUB}
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

    # model = models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    # 创建一个ResNet50模型（不使用预训练权重）
    model = models.resnet50(weights=None)
    # 将全连接层改为20个输出（对应VOC20类）
    model.fc = nn.Linear(2048, 20)
    # 加载指定的模型权重（权重文件中键为‘model’）
    model.load_state_dict(torch.load(args.resume, map_location='cpu')['model'])

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])

    # 目标层定义为ResNet50模型的layer4
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

    # 读取图像，将BGR（OpenCV读取格式为BGR）转换成RGB
    rgb_img = cv2.imread(args.image_path)[:, :, ::-1]
    # 输入图片尺寸调整为（224，224，3）
    rgb_img = cv2.resize(rgb_img, [224, 224])
    # 归一化图像像素值到[0,1]
    rgb_img = np.float32(rgb_img) / 255
    # 对图像进行预处理（减去均值，除以标准差），并转换为（B，C，H，W）。这个步骤通常包含归一化，我们修改了函数，使其能接收浮点图像，不需要再除以255进行归一化处理。
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [e.g ClassifierOutputTarget(281)]
    # 创建目标类别对象[ClassifierOutputTarget(8),ClassifierOutputTarget(10),ClassifierOutputTarget(14),ClassifierOutputTarget(15)]；并将输入张量复制多次（每个目标类别一次）input_tensor=[input_tensor,input_tensor,input_tensor,input_tensor]，在dim=0维度进行拼接，即由[1,3,224,224]-->[4,3,224,224]
    targets = [ClassifierOutputTarget(int(tar)) for tar in args.target]
    input_tensor = torch.cat([input_tensor for tar in args.target], dim=0)

    # 将模型设置成评估模式，并将模型跟输入张量转移到GPU运行
    model, input_tensor = model.eval().cuda(), input_tensor.cuda()
    # 进行推理，得到输出[4,20]（20个类别）
    out = model(input_tensor)
    # 取第一个样本的目标类别，用sigmoid计算每个类别的得分-->由logitc的输出转为[0,1]的概率值-->输出与args.target维度相同的数（组）
    y = torch.sigmoid(out)[0, args.target]
    print(y)

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    # 获得指定的CAM方法类
    cam_algorithm = methods[args.method]
    # with cam_algorithm(model=model,
    #                    target_layers=target_layers,
    #                    use_cuda=args.use_cuda) as cam:

    # 记录推理开始的时间
    start_time = time.time()

    # 创建CAM对象，model:ResNet; target_layers:解释的目标层； use_cuda：是否使用GPU
    cam = cam_algorithm(model=model, target_layers=target_layers, use_cuda=args.use_cuda)
    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    # 设置批量大小，用于Ablation CAM和Score CAM
    cam.batch_size = 32

    # 计算类激活图：input_tensor=[4,3,224,224]
    # 当我们在调用实例的时候就是先调用这个def __call__(self)方法
    grayscale_cams = cam(input_tensor=input_tensor,
                         targets=targets,
                         aug_smooth=args.aug_smooth,
                         eigen_smooth=args.eigen_smooth)

    # 因为input_tensor被复制4次（每个target一次），我们只取第一个样本
    out_first = out[0]
    # 使用sigmoid转换为概率 [20]
    probs = torch.sigmoid(out_first).detach().cpu().numpy()
    # 取Top - k类别，k与target中的参数数量相同
    topk = len(args.target)
    pred_classes = np.argsort(probs)[-topk:][::-1]

    print("Predicted classes:", pred_classes)
    print("Their scores:", probs[pred_classes])

    # 创建预测类别的target对象
    pred_targets = [ClassifierOutputTarget(int(tar)) for tar in pred_classes]

    # 仅针对预测类别重新构造输入张量，因为原来的input_tensor是按真实标签展开的
    input_tensor_pred = preprocess_image(rgb_img,
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    input_tensor_pred = torch.cat([input_tensor_pred for _ in pred_classes], dim=0).cuda()

    # 计算基于预测类别的CAM
    grayscale_cams_pred = cam(input_tensor=input_tensor_pred,
                              targets=pred_targets,
                              aug_smooth=args.aug_smooth,
                              eigen_smooth=args.eigen_smooth)

    # 计算并输出生成CAM所花费的时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("per time {}".format(total_time_str))

    layer = 0
    if target_layers == [model.layer4]:
        layer = 4
    elif target_layers == [model.layer3]:
        layer = 3
    elif target_layers == [model.layer2]:
        layer = 2
    elif target_layers == [model.layer2, model.layer3, model.layer4]:
        layer = 432
    elif layer == [model.layer3, model.layer4]:
        target_layers = 43
    elif layer == [model.layer2, model.layer4]:
        target_layers = 42

    # Here grayscale_cam has only one image in the batch
    # 将叠加了CAM热力图的图像显示出来。`show_cam_on_image`函数将热力图叠加到原始图像上。
    for index, grayscale_cam in enumerate(grayscale_cams):
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        show_image(cam_image)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.save_path, f'{args.method}_{layer}_{args.target[index]}.jpg'), cam_image)

    # 输出预测类别的可解释结果
    for index, grayscale_cam in enumerate(grayscale_cams_pred):
        cam_image_pred = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image_pred = cv2.cvtColor(cam_image_pred, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.save_path, f'{args.method}_{layer}_pred_{pred_classes[index]}.jpg'),
                    cam_image_pred)

    similarities = []
    for i in range(min(len(grayscale_cams), len(grayscale_cams_pred))):
        cam_true = grayscale_cams[i]
        cam_pred = grayscale_cams_pred[i]
        cam_true_flat = torch.tensor(cam_true.flatten(), dtype=torch.float32)
        cam_pred_flat = torch.tensor(cam_pred.flatten(), dtype=torch.float32)
        cos_sim = F.cosine_similarity(cam_true_flat.unsqueeze(0), cam_pred_flat.unsqueeze(0)).item()
        similarities.append(cos_sim)
    mean_cosine = np.mean(similarities)
    print(f"平均余弦相似度（真实标签 vs 预测标签）: {mean_cosine:.4f}")

    # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.

    # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    # gb = gb_model(input_tensor, target_category=args.target)
    #
    # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    # cam_gb = deprocess_image(cam_mask * gb)
    # gb = deprocess_image(gb)

    # if not os.path.exists(args.save_path):
    #     os.makedirs(args.save_path)
    # cv2.imwrite(os.path.join(args.save_path, f'{args.method}_cam.jpg'), cam_image)
    # cv2.imwrite(os.path.join(args.save_path, 'raw.jpg'), rgb_img[:, :, ::-1]*255)
    # cv2.imwrite(os.path.join(args.save_path, f'{args.method}_gb.jpg'), gb)
    # cv2.imwrite(os.path.join(args.save_path, f'{args.method}_cam_gb.jpg'), cam_gb)
