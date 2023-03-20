import cv2
import numpy as np
import torch
import math
import random
import time
# from openvino.inference_engine import IECore
# import onnxruntime
# import MNN.expr as F
# import seaborn as sns
# import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw, ImageFont
# import os
'''
——————————————————————————————————————————————Single frame infrared image segmentation————————————————————————————————
'''

# 通过cv2.dnn方式加载onnx框架
net = cv2.dnn.readNetFromONNX('results/LW_IRST.onnx')

# 通过openvino方式加载onnx框架
model = 'results/LW_IRST-sim.onnx'
ie = IECore()
net = ie.read_network(model=model)
input_blob = next(iter(net.input_info))
exec_net = ie.load_network(network=net, device_name="CPU")

# 通过onnxruntime方式加载onnx框架
model = 'results/LW_IRST.onnx'
session = onnxruntime.InferenceSession(model)

# 通过MNN方式加载加载mnn框架
mnnmodel = "results/LW_IRST.mnn"
vars = F.load_as_dict(mnnmodel)






