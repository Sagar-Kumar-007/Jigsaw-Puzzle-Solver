from bag_of_images import bag
import cv2
import numpy as np
from PIL import Image
import scipy
import glob
import numpy as np
import scipy.ndimage
import cv2
import skimage
#from skimage import img_as_ubyte
import matplotlib.pyplot as plt

def proj(src):


    bagOfImages=bag(src)
    complete_imagcopy=cv2.imread(src,0)


    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
        
    complete_imagcopy = skimage.util.img_as_ubyte(complete_imagcopy)
    kp2, des2 = sift.detectAndCompute(complete_imagcopy,None)

    bf = cv2.BFMatcher()
        
    for v in bagOfImages:
    #     plt.imshow(v)
    #     plt.show()
        
        # find the keypoints and descriptors with SIFT
        v= skimage.util.img_as_ubyte(v)
        kp1, des1 = sift.detectAndCompute(v,None)
        matches = bf.knnMatch(des1, des2, k=2)
        #matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        
        MIN_MATCH_COUNT = 10

        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h,w = v.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            # complete_imagcopy = cv2.polylines(complete_imagcopy,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        else:
            matchesMask = None
        draw_params = dict(matchColor = (0,255,0),singlePointColor = None,matchesMask = matchesMask, 
                    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #img3 = cv2.drawMatches(complete_imagcopy,kp1,v,kp2,good,None,**draw_params)
        img3 = cv2.drawMatches(v,kp1,complete_imagcopy,kp2,good,None,flags=2)
        
        #img3 = cv2.drawMatches(v,kp1,complete_imagcopy,kp2,good,None,**draw_params)
        plt.imshow(img3)
        plt.axis('off')
        plt.show()

    plt.imshow(complete_imagcopy,cmap = 'gray')
    plt.title("final image")
    plt.axis('off')
    plt.show()