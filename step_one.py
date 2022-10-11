import cv2 as cv
import requests
import base64
import os


def image_to_base64(image_np):
    image = cv.imencode('.jpg', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code

base_url = "http://127.0.0.1:5000"
process_url = base_url + "/process_img"
session = requests.session()

# cap = cv.VideoCapture(0)
# while True: # 获取摄像头
#     flag, img = cap.read()
#     if not flag:
#         break
#     img = cv2.flip(img, 1)
#     img_base64 = image_to_base64(img)
#     data = {'img_base64': img_base64}
#     resp = session.post(process_url, json=data)
#     cv2.waitKey(1)

to_process_path = 'to_process/2D'
to_process_imgs = os.listdir(to_process_path)
for img_name in to_process_imgs:
    img = cv.imread(os.path.join(to_process_path, img_name))
    img_base64 = image_to_base64(img)
    data = {'img_base64': img_base64}
    resp = session.post(process_url, json=data)

session.close()
