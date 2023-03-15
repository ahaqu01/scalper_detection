#!/bin/sh

# prepare base docker image
docker pull pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel 

# prepare pkgs
CRTDIR=$(pwd)
if [ ! -f "${CRTDIR}/pytorch-1.8.0-py3.7_cuda10.2_cudnn7.6.5_0.tar.bz2" ]; then
    wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/pytorch-1.8.0-py3.7_cuda10.2_cudnn7.6.5_0.tar.bz2 --no-check-certificate
fi

git clone --recurse-submodules https://gitlab.ictidei.cn/band-intel-center/Algorithm-platform/scalper_det.git huangniu_det
cd huangniu_det
git submodule update --init --recursive

cd ../
git clone https://github.com/open-mmlab/mmpose.git

# build docker image 
docker build -t huangniu_det_dockerfile_build:v1.0 .

# run container
docker run --gpus all -p 6688:6677 --shm-size=16g -itd --name huangniu_env huangniu_det_dockerfile_build:v1.0 /bin/bash /workspace/huangniu_det/start.sh
