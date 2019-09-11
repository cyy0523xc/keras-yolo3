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


def parse_input_image(pic='', pic_path='', image_type='jpg'):
    """人脸检测（输入的是base64编码的图像）
    :param pic 图片对象使用base64编码
    :param pic_path 图片路径
    :param image_type 输入图像类型, 取值jpg或者png
    :return image
    """
    if not pic and not pic_path:
        raise Exception('pic参数和pic_path参数必须有一个不为空')

    if pic:
        # 自动判断类型
        type_str = re.findall('^data:image/.+;base64,', pic)
        if len(type_str) > 0:
            if 'png' in type_str[0]:
                image_type = 'png'

        pic = re.sub('^data:image/.+;base64,', '', pic)
        pic = base64.b64decode(pic)
        pic = Image.open(io.BytesIO(pic))
        if image_type == 'png':   # 先转化为jpg
            bg = Image.new("RGB", pic.size, (255, 255, 255))
            bg.paste(pic, pic)
            pic = bg

        return pic

    return Image.open(pic_path)


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
