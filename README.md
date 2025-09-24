LW-IRSTNet: Lightweight Infrared Small Target Segmentation Network
https://doi.org/10.1109/TGRS.2023.3314586

https://patents.google.com/patent/CN116416430A

https://github.com/IRSTD/STD-EvalKit

Overview
LW-IRSTNet is a lightweight deep learning network designed for infrared small target segmentation, achieving state-of-the-art performance with extremely low computational complexity. The network has been patented (CN202310252896.1) and published in IEEE Transactions on Geoscience and Remote Sensing.

Key Features
​​High Accuracy​​: Achieves top-tier segmentation accuracy (mIoU, F1, ROC) on public datasets

​​Extremely Lightweight​​: Only 0.16M parameters and 303M FLOPs

​​Multi-Platform Deployment​​: Support for various mobile deployment formats

​​Real-Time Performance​​: 50FPS on Xiaomi Tablet 6Pro

Performance Highlights
Comparative experiments with 14 state-of-the-art networks demonstrate that LW-IRSTNet:

Matches or exceeds baseline results on all segmentation accuracy metrics (mIoU, F1, ROC)

Significantly reduces computational complexity compared to alternatives

Maintains robust performance across diverse datasets

Model Formats Available
We provide pre-trained models in multiple formats for easy deployment:

pkl(PyTorch)

onnx(Open Neural Network Exchange)

ncnn(Tencent NCNN)

tnn(Tencent TNN)

mnn(Alibaba MNN)

tflite(TensorFlow Lite)

Evaluation Metrics Toolkit
We developed ​​BinarySOSMetrics​​ (now integrated into STD-EvalKit) specifically for infrared small target segmentation evaluation:

Toolkit Features:
​​High Efficiency​​: Multi-threading support

​​Device Friendly​​: Automatic batch accumulation for all metrics

​​Unified API​​: Consistent interface across all metrics

Metric.update(labels, preds)for batch accumulation

Metric.get()to retrieve results

​​Unified Computation​​: Consistent algorithms ensuring result reproducibility

​​Flexible Input Support​​: Multiple data formats (hwc/chw/bchw/bhwc/image path)

Supported Metrics:
​​Pixel-Level​​: AUC ROC, AP PR, Precision, Recall, F1, IoU, NormalizedIoU

​​Center-Level​​: Precision, Recall, F1, Average Precision, Pd_Fa, ROC

​​Center Metrics​​: Normalized IoU, Mean Average Precision (COCO-style), Recall
