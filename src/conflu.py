# import packages
import cv2
import numpy as np
import matplotlib.pyplot as plt

import argparse


def run(args):
    # read the image
    img = cv2.imread(f'../data/input/{args.name}')

    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # gray scale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # binarization using Otsu's method
    _, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # configure the kernel
    kernel = np.ones((5, 5), np.uint8)
    
    # morphological transformation(Dilation)
    th_dilation = cv2.dilate(th, kernel, iterations=1)
    
    # contour extraction
    contours, _ = cv2.findContours(th_dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # draw the contours on the source image
    img_contour = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 2)
    
    # total number of pixels
    whole_area = th_dilation.size
    
    # number of zero area pixels
    white_area = cv2.countNonZero(th_dilation)
    
    # calculate confluency
    confluency = int(white_area / whole_area * 100)
    
    # show confluency
    print(f'Cell Confluency: {args.name} = {confluency} %')
    
    # visualization
    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    ax[0].imshow(img)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].text(1, 0.1, args.name, verticalalignment='top', color='red', size='x-large')
    ax[1].imshow(th_dilation, cmap='gray')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].imshow(img_contour)
    ax[2].text(1, 0.1, f'Cell Confluency: {confluency}', verticalalignment='top', color='red', size='x-large')
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    fig.savefig('../output/result.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    args = parser.parse_args()
    
    run(args)
