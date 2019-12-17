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
        'classes_path': 'model_data/coco_classes.txt',
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

# 获取目标类别
detect_classes = {}
for key, val in detect_configs.items():
    with open(val['classes_path']) as f:
        detect_classes[key] = [t.strip() for t in f.readlines()
                               if len(t.strip()) > 0]

# 预加载模型
yolo = YOLO(**detect_configs['card'])


def detect_images(filenames, classes=None):
    """检测多个图片
    :param filenames 文件名列表
    :param classes 需要检测的对象分类列表
    :return
    """
    res = []    # 保存结果数据

    for path in filenames:
        image_type = 'png' if path.lower().endswith('.png') else 'jpg'
        path = format_input_path(path)
        img = parse_input_image(image_path=path, image_type=image_type)
        _, data = yolo.detect_image(img)

        # 格式化返回类别值
        data['tags'] = format_classes(data['classes'], detect_classes['card'])

        if classes is not None:
            # 只保留需要的数据
            cond = np.array([i in classes for i in data['tags']])
        else:
            # 默认全部数据
            cond = np.array([True] * len(data['tags']))

        data['tags'] = np.array(data['tags'])
        res.append({
            'boxes': format_boxes(data['boxes'][cond].tolist()),
            'classes': data['tags'][cond].tolist(),
            'scores': data['scores'][cond].tolist(),
        })

    return res


def detect_image(image='', image_path='', image_type='jpg',
                 detect_type='common', return_img=False):
    """通用目标检测
    :param image 图片对象使用base64编码
    :param image_path 图片路径
    :param image_type 输入图像类型, 取值jpg或者png
    :return dict
    """
    if image_path != '':
        image_path = format_input_path(image_path)

    img = parse_input_image(image=image, image_path=image_path,
                            image_type=image_type)
    out_img, data = yolo.detect_image(img, out_img=True)
    return {
        'image': add_logo(out_img) if return_img else None,
        'boxes': data['boxes'].tolist(),
        'classes': format_classes(data['classes'],
                                  detect_classes[detect_type]),
        'scores': format_boxes(res['data']['boxes']),
    }


def get_demo_image(path):
    """获取演示图片"""
    img = Image.open(path)
    return {
        'image': parse_output_image(img)
    }


def format_classes(classes, config):
    """返回适合人类阅读的类别属性"""
    return [config[c] for c in classes]


def format_boxes(boxes):
    """boxes数据格式转化
    top, left, bottom, right ==> x, y, xb, yb
    """
    return [[left, top, right, bottom] for top, left, bottom, right in boxes]


if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'image':
        res = detect_image(image_path=sys.argv[2])
        print(res)
    elif sys.argv[1] == 'images':
        res = detect_images(sys.argv[2].split(';'))
        print(res)
