import cv2
import numpy as np

def padding(src):
    img = cv2.imread(src,0)
    h, w= img.shape
    maxi=max(h,w)

    height = maxi
    width = maxi
    dimensions = (width, height)
    new_image = cv2.resize(img, dimensions, interpolation=cv2.INTER_LINEAR)

    #ar=np.zeros((maxi,maxi))
    if (maxi%3==1):
        new_image = np.pad(new_image, pad_width=1, mode='constant', constant_values=0)
    elif (maxi%3==2):
        new_image = np.pad(new_image, pad_width=2, mode='constant', constant_values=0)

    return new_image