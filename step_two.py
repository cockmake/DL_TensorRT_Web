from DataType import DataType
import tensorrt as trt
import torch
from collections import OrderedDict, namedtuple
from flask import Flask, request, jsonify, json, Response
from flask_cors import CORS
import base64
import cv2 as cv
import numpy as np
from collections import deque

app = Flask(__name__)
CORS(app)

# 模型部分
confidence = 0.5
nms_threshold = 0.3
input_shape = (1, 3, 640, 640)
N, C, H, W = input_shape
logger = trt.Logger(trt.Logger.INFO)
device = torch.device('cuda:0')
Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
bindings = OrderedDict()
bindings_addrs = OrderedDict()
context, context2 = None, None


d_queue = deque(maxlen=100)

# 工具函数
def base64_to_image(base64_code):
    # base64解码
    img_data = base64.b64decode(base64_code)
    # 转换为np数组
    img_array = np.fromstring(img_data, np.uint8)
    # 转换成opencv可用格式
    img = cv.imdecode(img_array, cv.COLOR_RGB2BGR)
    return img



def format_img(np_img):

    _H, _W, _ = np_img.shape
    im = np.zeros((640, 640, 3), dtype=np.uint8)
    im[...] = 114
    factor_w = _W / 640
    factor_h = _H / 640
    factor = max(factor_w, factor_h)
    img = cv.resize(np_img, (int(_W / factor), int(_H / factor)))
    _H, _W, _ = img.shape
    dif_w = int((640 - _W) / 2)
    dif_h = int((640 - _H) / 2)
    im[dif_h: dif_h + _H, dif_w: dif_w + _W] = img


    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).float().to(device)  # float() 之关键  float() 之关键  float() 之关键  float() 之关键
    im /= 255  # 0 - 255 to 0.0 - 1.0
    return im, dif_w, dif_h, factor

# 加载模型
def load_model():
    global bindings_addrs, context, context2
    with open('2D_16.engine', 'rb') as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = model.get_binding_shape(index)
            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        bindings_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        context = model.create_execution_context()


def process_output(img_src, output_2d):
    src_h, src_w, _ = img_src.shape
    indices = cv.dnn.NMSBoxes(output_2d[:, :4], output_2d[:, 4], confidence, nms_threshold)
    boxes = output_2d[indices, :4]
    class_ids = output_2d[indices, 5]
    return boxes, class_ids


def draw_on_src(img_src, boxes, class_ids):
    for box in boxes:
        box = box.astype(np.int)
        cv.rectangle(img_src, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 3)

def image_to_base64(image_np):
    image = cv.imencode('.jpg', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code

def event_stream2():
    while True:
        if len(d_queue) != 0:
            dataType = d_queue.popleft()
            _H, _W, _ = dataType.img.shape
            img = cv.resize(dataType.img, (int(_W / 3), int(_H / 3)))
            img_base64 = image_to_base64(img)
            yield f"data: {json.dumps({'img_base64': img_base64, 'boxes': str(dataType.boxes / 3), 'id': str(dataType.class_id)})}\n\n"


@app.route('/get_data')
def stream2():
    return Response(event_stream2(), mimetype="text/event-stream")


i = 0
@app.route('/process_img', methods=['POST', 'GET'])
def process():
    global i
    data = request.json
    img_src = base64_to_image(data['img_base64'])
    img_blob, dif_w, dif_h, factor = format_img(img_src)
    bindings_addrs['images'] = img_blob.data_ptr()
    context.execute_v2(list(bindings_addrs.values()))
    out_prob = bindings['output0'].data.squeeze()
    out_prob = out_prob[out_prob[:, 4] > confidence]
    out_prob[:, 0] -= out_prob[:, 2] / 2 + dif_w
    out_prob[:, 1] -= out_prob[:, 3] / 2 + dif_h
    out_prob[:, :4] *= factor
    value, idx = torch.max(out_prob[:, 5:], dim=1)
    out_prob[:, 4] = value  # 写入confidence
    out_prob[:, 5] = idx
    out_prob = out_prob[:, :6].cpu().numpy()
    boxes, class_ids = process_output(img_src, out_prob)
    draw_on_src(img_src, boxes, class_ids)   # 可以开线程加入队列  节省处理时间
    if len(d_queue) != d_queue.maxlen:
        d_queue.append(DataType(img_src, boxes, class_ids))
    # cv.imwrite(os.path.join('imgs', str(i) + '.jpg'), img_src)

    i += 1
    return jsonify({'ret': '处理完毕, 准备下一批的处理'})

load_model()
if __name__ == '__main__':
    app.run(host='0.0.0.0')



