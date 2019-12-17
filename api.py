# -*- coding: utf-8 -*-
#
#
# Author: alex
# Created Time: 2019年09月11日 星期三 18时12分55秒
from fireRest import API, app
from server import detect_images, detect_image, get_demo_image, detect_b64s

API(detect_images)
API(detect_b64s)
API(detect_image)
API(get_demo_image)
app.run(port=20920, host='0.0.0.0', debug=True)
