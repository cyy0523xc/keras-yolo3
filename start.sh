#!/bin/bash
# 
# 启动
# ./start.sh registry.cn-hangzhou.aliyuncs.com/ibbd/video:cu100-py35-u1604-cv-tf-pytorch \
#     python3 server.py
# Author: alex
# Created Time: 2019年09月12日 星期四 10时13分07秒
cmd=$*
if [ $# -le 2 ]; then
    cmd="$* /bin/bash"
fi
echo "Command: $cmd"

docker rm -f ibbd-yolov3
docker run --rm -d --runtime=nvidia --name ibbd-yolov3 \
    -p 20920:20920 \
    -v `pwd`:/yolov3 \
    -e PYTHONIOENCODING=utf-8 \
    -w /yolov3 \
    $cmd
