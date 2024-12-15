from cp_hw6 import computeIntrinsic, computeExtrinsic, pixel2ray
import os
import glob
import numpy as np
import cv2
from triangulate import getIMGPaths, getImageStack, classify_imgstack_codes, decode_gray, read_img, get_per_pixel_threshold
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #Example for running intrinsic and extrinsic calibration on provided structured light examples

    #Input data locations
    #for CAMERA calibration:
    baseDir = './data/calib' #data directory
    calName = 'normal00013.JPG' #calibration sourse (also a dir in data)
    
    useLowRes = False #enable lowres for debugging

    
    w = 1719
    h = 2128
    #2235 x, 943 y

    start_h = 943
    start_w = 2235

    def get_slice(img):
        return img[start_h: start_h+h, start_w:start_w+w, :]



    #Extrinsic calibration parameters
    dW1 = (8, 8) #window size for finding checkerboard corners
    checkerboard = (6, 8) #number of internal corners on checkerboard
    size_of_square = 0.0235 #in cm

    
    ################################################
    ################################################
    ################################################
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[0,:,:2] = (size_of_square*np.mgrid[0:checkerboard[0], 0:checkerboard[1]]).T.reshape(-1, 2)
    img_shape = None


    TEST_IMG = os.path.join(baseDir, calName)
    print("TEST IMG PATH", TEST_IMG)
    img = cv2.imread(TEST_IMG)
    img = get_slice(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_shape = gray.shape[::-1]
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    # TODO: touchup
    ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        #print(corners)
        corners2 = cv2.cornerSubPix(gray, corners, dW1, (-1,-1), criteria)

        imgpoints.append(corners2)  
    else:
        print("could not get checkerboard")
        #resized_image = cv2.resize(img, (800, 600))
        cv2.imshow('img',img)
        cv2.setWindowTitle('img', "final image of intr calibration, corner mapping") 
        cv2.waitKey(0)        
        exit()
          

    
    print(f"{len(objpoints)}obj points are", objpoints[0].shape)
    print(f"{len(imgpoints)}img points are", imgpoints)


    img = cv2.drawChessboardCorners(img, checkerboard, corners2, ret)
    resized_image = cv2.resize(img, (800, 600))
    cv2.imshow('img',resized_image)
    cv2.setWindowTitle('img', "final image of intr calibration, corner mapping") 
    cv2.waitKey(0)

    # plt.imshow(img)
    # plt.show()
