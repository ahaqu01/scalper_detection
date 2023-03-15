#!/bin/sh

docker stop huangniu_env
docker rm huangniu_env

docker run --gpus all -p 6688:6677 --shm-size=16g -itd --name huangniu_env huangniu_det_dockerfile_build:v1.0 /bin/bash /workspace/huangniu_det/start.sh
