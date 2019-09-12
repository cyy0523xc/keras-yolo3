# -*- coding: utf-8 -*-
#
#
# Author: alex
# Created Time: 2019年09月11日 星期三 18时12分55秒
from PIL import Image
from yolo import YOLO
from utils import parse_input_image, parse_output_image, add_logo


common_config = {
    'model_path': 'model_data/yolov3-spp.h5'
}


def common_image(pic='', pic_path='', image_type='jpg',):
    """通用目标检测
    :param pic 图片对象使用base64编码
    :param pic_path 图片路径
    :param image_type 输入图像类型, 取值jpg或者png
    :return dict
    """
    img = parse_input_image(pic=pic, pic_path=pic_path, image_type=image_type)
    yolo = YOLO(**common_config)
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


def get_common_image(path):
    """获取演示图片"""
    img = Image.open(path)
    return {
        'image': parse_output_image(img)
    }


if __name__ == '__main__':
    from fireRest import API, app
    API(common_image)
    app.run(port=20960, host='0.0.0.0', debug=True)
