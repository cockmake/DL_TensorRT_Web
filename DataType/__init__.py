from collections import deque
import threading
import cv2 as cv
from flask import json
from utils import base64_to_image, image_to_base64
class DataType:
    def __init__(self, img, boxes, class_id):
        self.img = img
        self.boxes = boxes
        self.class_id = class_id


class SendFrame:
    def __init__(self, maxlen=100):
        self.d_queue = deque(maxlen=maxlen)

    def frame(self):
        while True:
            if len(self.d_queue) != 0:
                dataType = self.d_queue.popleft()
                _H, _W, _ = dataType.img.shape
                img = cv.resize(dataType.img, (788, 394))
                img_base64 = image_to_base64(img)
                yield f"data: {json.dumps({'img_base64': img_base64, 'boxes': str(dataType.boxes / 1.3), 'id': str(dataType.class_id)})}\n\n"
            else:
                yield f"data {json.dumps({'ret': 'null'})}\n\n"


    def append(self, dataType):
        if len(self.d_queue) != self.d_queue.maxlen:
            self.d_queue.append(dataType)


sf = SendFrame()