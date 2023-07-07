import torch
import cv2
import numpy as np


def get_model(name):
    if name == 'resnet':
        from .basic_embedding import ResNetFeats
        return ResNetFeats
    elif name == 'vgg':
        from .basic_embedding import VGGFeats
        return VGGFeats
    else:
        raise NotImplementedError


def return_CAM(feature_conv, weight, H=224, W=224):
    # generate the class -activation maps upsample to (224, 224)
    size_upsample = (H, W)
    bz, t, nc, h, w = feature_conv.shape
    output_cam = []

    # beforeDot = feature_conv.reshape((nc, h*w))
    # cam = np.matmul(weight, beforeDot)

    cam = np.mean(feature_conv[0, 0, :, :], axis=0)

    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))

    return output_cam


def return_activation_map(model, features, images, layer_name):

    if layer_name != None:
        for name, param in model.named_parameters():
            # get the weights of the last layer
            if name == layer_name:
                weights = param.data.cpu().numpy()
    else:
        weights = None

    _, _, _, H, W = images.shape

    feat_cam = features.cpu().numpy()
    CAMs = return_CAM(feat_cam, weights, H, W)
    _, _, _, height, width = images.shape
    heatmap = cv2.applyColorMap(cv2.resize(
        CAMs[0], (width, height)), cv2.COLORMAP_JET)
    image = np.transpose(images.cpu().numpy()[
        0, 0, :, :, :]*255, (1, 2, 0))

    return heatmap * 0.5 + image * 0.5
