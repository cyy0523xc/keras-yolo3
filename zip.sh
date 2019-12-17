#!/bin/bash
# 
# 将项目打包压缩
# Author: alex
# Created Time: 2019年03月11日 星期一 10时38分16秒
cd ../
if [ ! -d keras-yolo3 ]; then
    echo "$PWD: 当前目录错误."
fi
version=
if [ $# = 1 ]; then 
    version="-$1"
fi

date_str=`date -I`
filename=keras-yolo3-"${date_str//-/}$version".zip
if [ -f "$filename" ]; then
    rm -f "$filename"
fi

zip -r "$filename" keras-yolo3 \
    -x "*/.git/*" \
    -x "*/.*" \
    -x "*/*/*.swp" \
    -x "*/__pycache__/*" 
