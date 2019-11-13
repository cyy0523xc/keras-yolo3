# -*- coding: utf-8 -*-
#
#
# Author: alex
# Created Time: 2019年09月09日 星期一 15时51分40秒
import re
import io
import cv2
import base64
from PIL import Image
import numpy as np
from os import environ
from os.path import join, isfile, isdir

# 可以通过环境变量来指定跟目录
if 'VIDEO_ROOT_PATH' in environ:
    root_path = environ['VIDEO_ROOT_PATH']
    if not root_path.endswith('/'):
        root_path += '/'
else:
    root_path = '/data/'


def format_input_path(path):
    """格式化输入路径"""
    full_path = join(root_path, path)
    if isfile(full_path) or isdir(full_path):
        return full_path
    raise Exception('错误的路径：' + path)


def format_output_path(path):
    """格式化输出的文件地址"""
    if path.startswith(root_path):
        return path[len(root_path):]
    return path


def parse_input_image(image='', image_path='', image_type='jpg'):
    """人脸检测（输入的是base64编码的图像）
    :param image 图片对象使用base64编码
    :param image_path 图片路径
    :param image_type 输入图像类型, 取值jpg或者png
    :return image
    """
    if not image and not image_path:
        raise Exception('image参数和image_path参数必须有一个不为空')

    if image:
        # 自动判断类型
        type_str = re.findall('^data:image/.+;base64,', image)
        if len(type_str) > 0:
            if 'png' in type_str[0]:
                image_type = 'png'

        image = re.sub('^data:image/.+;base64,', '', image)
        image = base64.b64decode(image)
        image = Image.open(io.BytesIO(image))
        if image_type == 'png':   # 先转化为jpg
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, image)
            image = bg

        return image

    return Image.open(image_path)


def parse_output_image(out_img):
    """cv2转base64字符串"""
    output_buffer = io.BytesIO()
    out_img.save(output_buffer, format='JPEG')
    binary_data = output_buffer.getvalue()
    return str(base64.b64encode(binary_data), encoding='utf8')


def add_logo(img):
    """增加公司logo"""
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    h, w, _ = img.shape
    cv2.putText(img, 'DeeAo AI Team', (w-250, h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 255, 0), 2)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
