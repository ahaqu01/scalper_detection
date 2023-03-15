FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel
MAINTAINER yuyingchao

WORKDIR /envs/

#EXPOSE 6677

RUN rm -r /workspace \
    && mkdir /workspace

COPY ./pytorch-1.8.0-py3.7_cuda10.2_cudnn7.6.5_0.tar.bz2 /envs/pytorch-1.8.0-py3.7_cuda10.2_cudnn7.6.5_0.tar.bz2
COPY ./mmpose  /envs/mmpose
COPY ./huangniu_det  /workspace/huangniu_det

RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get install -y git \
    && apt-get install -y vim 

# install base requirements
RUN pip install numpy==1.20.2 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install Flask==2.0.2 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install Flask-Cors==3.0.10 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install gevent==21.12.0 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install h5py \
    && pip install thop==0.0.31.post2005241907 -i https://pypi.tuna.tsinghua.edu.cn/simple \ 
    && pip install PyYAML==5.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install opencv-contrib-python==4.5.4.60 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install opencv-python==4.5.4.60 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install opencv-python-headless==4.5.4.60 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install scikit-learn==1.0.1 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install Pillow==9.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install Cython==0.29.26 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install cython_bbox \
    && pip install lap==0.4.0 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install scipy==1.7.2 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install scikit_image==0.19.1 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install matplotlib==3.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install torchvision==0.9.1 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && conda install --use-local pytorch-1.8.0-py3.7_cuda10.2_cudnn7.6.5_0.tar.bz2

RUN pip install mmcv-full==1.4.1 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html

RUN cd /envs/mmpose \
    && pip install -r requirements/build.txt \
    && pip install -r requirements/runtime.txt \
    && pip install -r requirements/tests.txt \
    && pip install -v -e . 

