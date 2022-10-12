import base64
import numpy as np
import cv2 as cv
# 工具函数
def base64_to_image(base64_code):
    # base64解码
    img_data = base64.b64decode(base64_code)
    # 转换为np数组
    img_array = np.fromstring(img_data, np.uint8)
    # 转换成opencv可用格式
    img = cv.imdecode(img_array, cv.COLOR_RGB2BGR)
    return img

def image_to_base64(image_np):
    image = cv.imencode('.jpg', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code