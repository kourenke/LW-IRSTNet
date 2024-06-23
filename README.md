# LW-IRSTNet

专利保护：申请(专利)号：	CN202310252896.1； 申请公布号：	CN116416430A

R. Kou et al., "LW-IRSTNet: Lightweight Infrared Small Target Segmentation Network and Application Deployment," in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-13, 2023, Art no. 5621313, doi: 10.1109/TGRS.2023.3314586. 

To verify the accuracy, robustness, and computational complexity of LW-IRSTNet, 14 state-of-the-art networks are used as baselines for comparative analysis. The experimental results showed that the segmentation accuracy indexes (mIOU, F1, ROC) of LW-IRSTNet are all above or equal to the baseline best results on the public datasets. Meanwhile, the network params are compressed to 0.16M and FLOPs to 303M, which is much lower than the baseline results.

To deploy LW-IRSTNet on different mobile terminals, I uploaded models in different formats, including pkl, onnx, ncnn, tnn, and mnn.

I recently uploaded a. tflite format model. This model was deployed on the Xiaomi tablet 6Pro and achieved a good performance of 50FPS.

In addition, we have also compiled a set of evaluation index libraries suitable for algorithms in this field, named BinarySOSMetrics.

The relevant code is published on https://github.com/BinarySOS/BinarySOSMetrics.

The main features of BinarySOSMetrics include:

High Efficiency: Multi-threading.

Device Friendly: All metrics support automatic batch accumulation.

Unified API: All metrics provide the same API, Metric.update(labels, preds) complete the accumulation of batches， Metric.get() get metrics。

Unified Computational: We use the same calculation logic and algorithms for the same type of metrics, ensuring consistency between results.

Supports multiple data formats: Supports multiple input data formats, hwc/chw/bchw/bhwc/image path, more details in ./notebook/tutorial.ipynb


