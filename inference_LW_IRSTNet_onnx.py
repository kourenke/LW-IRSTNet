import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time


'''
——————————————————————————————————————————————Single frame infrared image segmentation——————————————————————————————————————————————————————————————
'''
def getZ(img,X,Y):
    gray = img[X,Y]
    return gray

def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    preprocessed_img = preprocessed_img.numpy()
    return preprocessed_img


if __name__ == '__main__':
    # load network
    net = cv2.dnn.readNetFromONNX('results/LW_IRST.onnx')
    # load image
    base_size = 256
    img = cv2.imread('./data/single/2.png', 1)
    img = np.float32(cv2.resize(img, (base_size, base_size))) / 255
    input = preprocess_image(img)
    net.setInput(input)
    # inference in cpu
    print('...inference in progress')
    start = time.perf_counter()
    output = net.forward()
    output1 = output.reshape(base_size, base_size)
    output2 = output1>0
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Input image')
    plt.subplot(122), plt.imshow(output2,cmap='gray'), plt.title('Inference result')
    plt.show()


