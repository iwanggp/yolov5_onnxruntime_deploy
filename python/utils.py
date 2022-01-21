#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@File   :utils.py
@IDE    :PyCharm
@Author :gpwang
@Date   :2022/1/21
@Desc   :yolov5的一些处理工具
=================================================='''
import numpy as np
import cv2


def letterbox(img, new_shape=(640, 640), auto=False, scaleFill=False, scaleUp=True):
    """
    python的信封图片缩放
    :param img: 原图
    :param new_shape: 缩放后的图片
    :param color: 填充的颜色
    :param auto: 是否为自动
    :param scaleFill: 填充
    :param scaleUp: 向上填充
    :return:
    """
    shape = img.shape[:2]  # current shape[height,width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleUp:
        r = min(r, 1.0)  # 确保不超过1
    ration = r, r  # width,height 缩放比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ration = new_shape[1] / shape[1], new_shape[0] / shape[0]
    # 均分处理
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # 添加边界
    return img, ration, (dw, dh)


def clip_coords(boxes, img_shape):
    """
    图片的边界处理
    :param boxes: 检测框
    :param img_shape: 图片的尺寸
    :return:
    """
    boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3].clip(0, img_shape[0])  # x2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    坐标还原
    :param img1_shape: 旧图像的尺寸
    :param coords: 坐标
    :param img0_shape:新图像的尺寸
    :param ratio_pad: 填充率
    :return:
    """
    if ratio_pad is None:  # 从img0_shape中计算
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain=old/new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords
