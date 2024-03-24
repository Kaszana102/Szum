import os

import cv2
import matplotlib.pyplot as plt



def directory_histogram(directory):
    hist_0 = [[0]]*256
    hist_1 = [[0]]*256
    hist_2 = [[0]]*256
    for filename in os.listdir(directory):
        # load image
        img = cv2.imread(directory+'/'+filename)


        hist_0 += cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_1 += cv2.calcHist([img], [1], None, [256], [0, 256])
        hist_2 += cv2.calcHist([img], [2], None, [256], [0, 256])
        print(filename)
    plt.figure()
    hist_0/= max(hist_0)
    hist_1/= max(hist_1)
    hist_2/= max(hist_2)
    plt.plot([val[0] for val in hist_0], color='b')
    plt.plot([val[0] for val in hist_1], color='g')
    plt.plot([val[0] for val in hist_2], color='r')
    plt.xlim([0, 256])
    plt.show()


<<<<<<< Updated upstream

directory_histogram('dataset_src/horses')
=======
directory_histogram('dataset/penguins')
>>>>>>> Stashed changes
