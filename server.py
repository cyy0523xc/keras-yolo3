# -*- coding: utf-8 -*-
#
#
# Author: alex
# Created Time: 2019年09月11日 星期三 18时12分55秒
from PIL import Image
from yolo import YOLO
from utils import parse_input_image, parse_output_image, \
    format_input_path

# 识别模型配置
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

# 定义启动使用的模型
detect_config = detect_configs['card']

# 获取目标类别
detect_classes = []
with open(detect_config['classes_path']) as f:
    detect_classes = [t.strip() for t in f.readlines()
                      if len(t.strip()) > 0]

# 预加载模型
yolo = YOLO(**detect_config)


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
        row = {
            'bboxes': [],
            'classes': [],
            'scores': [],
        }
        if data['bboxes'] is None:
            res.append(row)
            continue

        data['classes'] = format_classes(data['classes'])
        for box, cs, score in zip(data['bboxes'], data['classes'], data['scores']):
            if cs not in classes:
                continue
            row['bboxes'].append(box)
            row['classes'].append(cs)
            row['scores'].append(score)

        res.append(row)

    return res


def detect_b64s(b64_list, classes=None):
    """检测多个图片
    :param b64_list 文件base64列表
    :param classes 需要检测的对象分类列表
    :return
    """
    res = []    # 保存结果数据
    images = [parse_input_image(image=b64) for b64 in b64_list]
    for img in images:
        _, data = yolo.detect_image(img)
        row = {
            'bboxes': [],
            'classes': [],
            'scores': [],
        }
        if data['bboxes'] is None:
            res.append(row)
            continue

        data['classes'] = format_classes(data['classes'])
        for box, cs, score in zip(data['bboxes'], data['classes'], data['scores']):
            if cs not in classes:
                continue
            row['bboxes'].append(box)
            row['classes'].append(cs)
            row['scores'].append(score)

        res.append(row)

    return res


def detect_image(image='', image_path='', image_type='jpg', return_img=False):
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
    if data['bboxes'] is None:
        return {
            'bboxes': [],
            'classes': [],
            'scores': [],
        }

    return {
        'image': parse_output_image(out_img) if return_img else None,
        'bboxes': format_bboxes(data['bboxes'].tolist()),
        'scores': data['scores'].tolist(),
        'classes': format_classes(data['classes'].tolist()),
    }


def get_demo_image(path):
    """获取演示图片"""
    img = Image.open(path)
    return {
        'image': parse_output_image(img)
    }


def format_classes(classes):
    """返回适合人类阅读的类别属性"""
    return [detect_classes[c] for c in classes]


def format_bboxes(bboxes):
    """bboxes数据格式转化
    top, left, bottom, right ==> x, y, xb, yb
    """
    return [[left, top, right, bottom] for top, left, bottom, right in bboxes]


if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'image':
        res = detect_image(image_path=sys.argv[2])
        print(res)
    elif sys.argv[1] == 'images':
        res = detect_images(sys.argv[2].split(';'))
        print(res)