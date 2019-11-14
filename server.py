# -*- coding: utf-8 -*-
#
#
# Author: alex
# Created Time: 2019年09月11日 星期三 18时12分55秒
import numpy as np
from PIL import Image
from yolo import YOLO
from utils import parse_input_image, parse_output_image, add_logo, \
    format_input_path


detect_configs = {
    # 通用目标检测
    'common': {
        'model_path': 'model_data/yolov3-spp.h5',
    },
    # GF Logo&证件识别
    'card': {
        'model_path': 'model_data/gf_yolov3_spp_l066425.h5',
        'classes_path': 'model_data/gf_yolov3_spp_classes.txt',
    },
    # 安全帽工作服识别
    'helmet': {
        'model_path': 'model_data/yolov3_shantou_0512.h5',
        'classes_path': 'model_data/yolov3_shantou_classes.txt',
    }
}


def detect_images(filenames, classes=None):
    """检测多个图片
    :param filenames 文件名列表
    :param classes 需要检测的对象分类列表
    """
    res = []    # 保存结果数据
    yolo = YOLO(**detect_configs['card'])

    for path in filenames:
        image_type = 'png' if path.lower().endswith('.png') else 'jpg'
        path = format_input_path(path)
        img = parse_input_image(image_path=path, image_type=image_type)
        out_img, data = yolo.detect_image(img)

        if classes is not None:
            # 只保留需要的数据
            cond = np.array([i in classes for i in data['classes']])
        else:
            # 默认全部数据
            cond = np.array([True] * len(data['classes']))

        res.append({
            'boxes': data['boxes'][cond].tolist(),
            'classes': data['classes'][cond].tolist(),
            'scores': data['scores'][cond].tolist(),
        })

    yolo.close_session()
    return res


def detect_image(image='', image_path='', image_type='jpg',
                 detect_type='common'):
    """通用目标检测
    :param image 图片对象使用base64编码
    :param image_path 图片路径
    :param image_type 输入图像类型, 取值jpg或者png
    :return dict
    """
    return do_detect_image(detect_configs[detect_type], image=image,
                           image_path=image_path, image_type=image_type)


def do_detect_image(detect_cfg, image='', image_path='', image_type='jpg'):
    """目标检测
    :param detect_cfg 检测配置
    :param image 图片对象使用base64编码
    :param image_path 图片路径
    :param image_type 输入图像类型, 取值jpg或者png
    :return dict
    """
    img = parse_input_image(image=image, image_path=image_path,
                            image_type=image_type)
    yolo = YOLO(**detect_cfg)
    out_img, data = yolo.detect_image(img)
    yolo.close_session()

    out_img = add_logo(out_img)
    return {
        'image': parse_output_image(out_img),
        'data': {
            'boxes': data['boxes'].tolist(),
            'classes': data['classes'].tolist(),
            'scores': data['scores'].tolist(),
        }
    }


def get_demo_image(path):
    """获取演示图片"""
    img = Image.open(path)
    return {
        'image': parse_output_image(img)
    }


if __name__ == '__main__':
    from fireRest import API, app
    API(detect_images)
    API(detect_image)
    API(get_demo_image)
    app.run(port=20920, host='0.0.0.0', debug=True)
