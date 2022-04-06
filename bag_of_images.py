import cv2
import numpy as np
import random
from img_padding import padding

def bag(src):
    img=padding(src)
    bag=[]
    m,n=img.shape
    x=m//3
    bag.append(img[0:x,0:x])
    bag.append(img[x:2*x,0:x])
    bag.append(img[2*x:3*x,0:x])
    
    bag.append(img[0:x,x:2*x])
    bag.append(img[x:2*x,x:2*x])
    bag.append(img[2*x:3*x,x:2*x])
    
    bag.append(img[0:x,2*x:3*x])
    bag.append(img[x:2*x,2*x:3*x])
    bag.append(img[2*x:3*x,2*x:3*x])
    
    # cv2.imshow("safdas",bag[0])
    # cv2.imshow("safdasdas",bag[1])
    # cv2.waitKey(0)

    random.shuffle(bag)

    # for i in bag:
    #     cv2.imshow('image',i)

    shuffled_img=np.zeros((m,n)).astype('uint8')
    shuffled_img[0:m//3,0:m//3]=bag[0]
    shuffled_img[m//3:2*(m//3),0:m//3]=bag[1]
    shuffled_img[2*(m//3):m,0:m//3]=bag[2]

    shuffled_img[0:m//3,m//3:2*(m//3)]=bag[3]
    shuffled_img[m//3:2*(m//3),m//3:2*(m//3)]=bag[4]
    shuffled_img[2*(m//3):m,m//3:2*(m//3)]=bag[5]

    shuffled_img[0:m//3,2*(m//3):m]=bag[6]
    shuffled_img[m//3:2*(m//3),2*(m//3):m]=bag[7]
    shuffled_img[2*(m//3):m,2*(m//3):m]=bag[8]
    cv2.imshow('org_image',img)
    cv2.imshow('new_image',shuffled_img)
    cv2.waitKey(0)
    cv2. destroyAllWindows()
    return bag