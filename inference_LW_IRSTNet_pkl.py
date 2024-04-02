import torch
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import cv2
from  mpl_toolkits.mplot3d import Axes3D
from thop import profile
import time

def getZ(img,X,Y):
    gray = img[X,Y]
    return gray


def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Inference of Net')

    #
    # Checkpoint parameters
    #
    parser.add_argument('--pkl-path', type=str, default=r'./results/merged_LW_IRST_ablation_Iter-18400_mIoU-0.6638_fmeasure-0.7979.pkl',
                        help='checkpoint path')

    #
    # Test image parameters
    #
    parser.add_argument('--image-path', type=str, default=r'./data/single/34.jpg', help='image path')
    parser.add_argument('--base-size', type=int, default=256, help='base size of images')


    args = parser.parse_args()
    return args


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
    input = Variable(preprocessed_img, requires_grad = True)
    return input


if __name__ == '__main__':
    args = parse_args()

    # load network
    print('...load checkpoint: %s' % args.pkl_path)
    net = torch.load(args.pkl_path, map_location=torch.device('cpu'))
    net.eval()

    # load image
    print('...loading test image: %s' % args.image_path)
    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (args.base_size, args.base_size))) / 255
    input = preprocess_image(img)

    # inference in cpu
    print('...inference in progress')
    start = time.perf_counter()
    with torch.no_grad():
        output = net(input)   #(1,1,256,256)
    output = output.cpu().detach().numpy().reshape(args.base_size, args.base_size)>0    #(256,256)
    output = output.astype(np.uint8) * 255  # (256,256)

    # figure1 =plt.figure()
    plt.subplot(121), plt.imshow(img,cmap='gray')
    plt.subplot(122), plt.imshow(output,cmap='gray')
    plt.show()