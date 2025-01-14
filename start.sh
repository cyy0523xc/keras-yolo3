#!/bin/bash
# 
# 启动
# ./start.sh registry.cn-hangzhou.aliyuncs.com/ibbd/video:cu100-py35-u1604-cv-tf-pytorch \
#     python3 api.py
# 转换: python3 convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
# Author: alex
# Created Time: 2019年09月12日 星期四 10时13分07秒
cmd=$*
args="-d"
if [ $# -le 2 ]; then
    cmd="$* /bin/bash"
    args="-ti"
fi
echo "Command: $cmd"

docker rm -f ibbd-yolov3
docker run --rm "$args" --runtime=nvidia --name ibbd-yolov3 \
    --volumes-from ibbd-data \
    -v /etc/timezone:/etc/timezone \
    -v /etc/localtime:/etc/localtime \
    -v `pwd`:/yolov3 \
    -e PYTHONIOENCODING=utf-8 \
    -p 20922:20920 \
    -w /yolov3 \
    $cmd
