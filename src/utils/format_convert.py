import cv2
import numpy as np
import base64

def numpy_to_base64(image_np):
    '''numpy 转 base64'''
    data = cv2.imencode('.jpg', image_np)[1]
    image_bytes = data.tobytes()
    image_base4 = base64.b64encode(image_bytes).decode('utf8')
    return image_base4

def numpy_to_bytes(image_np):
    '''numpy 转 bytes'''
    data = cv2.imencode('.jpg', image_np)[1]
    image_bytes = data.tobytes()
    return image_bytes

# 数组保存
def numpy_to_file(image_np):
    filename = '你的文件名_numpy.jpg'
    cv2.imwrite(filename,image_np)
    return filename

# bytes转数组
def bytes_to_numpy(image_bytes):
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image_np2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image_np2

# bytes 转 base64
def bytes_to_base64(image_bytes):
    image_base4 = base64.b64encode(image_bytes).decode('utf8')
    return image_base4

# bytes 保存
def bytes_to_file(image_bytes):
    filename = '你的文件名_bytes.jpg'
    with open(filename,'wb') as f:
        f.write(image_bytes)
        return filename

# 文件 转 数组
def file_to_numpy(path_file):
    image_np = cv2.imread(path_file)
    return image_np

# 文件转 字节
def file_to_bytes(path_file):
    with open(path_file,'rb') as f:
        image_bytes = f.read()
        return image_bytes

# 文件转base64
def file_to_base64(path_file):
    with open(path_file,'rb') as f:
        image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf8')
        return image_base64

# base64 转 bytes
def base64_to_bytes(image_base64):
    image_bytes = base64.b64decode(image_base64)
    return image_bytes

# base64转数组
def base64_to_numpy(image_base64):
    image_bytes = base64.b64decode(image_base64)
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image_np2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image_np2

# base64 保存
def base64_to_file(image_base64):
    filename = '你的文件名_base64.jpg'
    image_bytes = base64.b64decode(image_base64)
    with open(filename, 'wb') as f:
        f.write(image_bytes)
        return filename