import os

import cv2
import matplotlib.pyplot as plt



def directory_histogram(directory):

    for filename in os.listdir(directory):
        # load image
        img = cv2.imread(directory+'/'+filename)

        avg_hist = [[0,0,0]]*256

        color = ('b', 'g', 'r')
        hist_color = [[0,0,0]]
        histr = cv2.calcHist([img], [0,1], None, [256], [0, 256])
        hist_color += histr

    plt.figure()
    avg_hist/= len(os.listdir(directory))
    plt.plot([val[0] for val in avg_hist], color='r')
    plt.plot([val[0] for val in avg_hist], color='g')
    plt.plot([val[0] for val in avg_hist], color='b')
    plt.xlim([0, 256])
    plt.show()



directory_histogram('dataset/horses')