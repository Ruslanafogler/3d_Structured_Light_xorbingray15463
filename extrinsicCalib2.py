from cp_hw6 import computeIntrinsic, computeExtrinsic, pixel2ray
import os
import glob
import numpy as np
import cv2
from util import getIMGPaths, getImageStack, classify_imgstack_codes, decode_gray, read_img, get_per_pixel_threshold
import matplotlib.pyplot as plt  



criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.001)

def get_chessboard_imgpts(img, checkerboard, dW1):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)
    if ret == True:
        corners = cv2.cornerSubPix(gray, corners, dW1, (-1,-1), criteria)
        img = cv2.drawChessboardCorners(img, checkerboard, corners, ret)
    else:
        print("error: checkerboard not found")
    resized_image = cv2.resize(img, (800, 600))
    cv2.imshow('img',resized_image)
    cv2.setWindowTitle('img', "chessboard") 
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    return corners



def get_extrinsic(objp, checkerboard, dW1, mtx, dist, img_path):

    corners2 = get_chessboard_imgpts(cv2.imread(images[0]), checkerboard, dW1)
    ret, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
    R = cv2.Rodrigues(rvec)[0]
    T = tvec.ravel()

    print("EXTRINSIC PARAMETERS:")
    print("matrix: \n")
    print(R)
    print("Translation: \n")
    print(T) 

    color_img = cv2.imread(img_path)
    
    axis = np.float32([[0,0,0], [1,0,0], [0,1,0], [0,0,1]]).reshape(-1,3)
    axis_img = cv2.projectPoints(axis, rvec, tvec, mtx, dist)[0]
    axis_img = axis_img.astype(int)
    #cv2.namedWindow('Result')
    cv2.putText(color_img, 'X', (axis_img[1,0,0], axis_img[1,0,1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255))
    cv2.line(color_img, (axis_img[0,0,0], axis_img[0,0,1]), (axis_img[1,0,0], axis_img[1,0,1]), (0,0,255), 2) #x
    cv2.putText(color_img, 'Y', (axis_img[2,0,0], axis_img[2,0,1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))
    cv2.line(color_img, (axis_img[0,0,0], axis_img[0,0,1]), (axis_img[2,0,0], axis_img[2,0,1]), (0,255,0), 2) #y
    cv2.putText(color_img, 'Z', (axis_img[3,0,0], axis_img[3,0,1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0))
    cv2.line(color_img, (axis_img[0,0,0], axis_img[0,0,1]), (axis_img[3,0,0], axis_img[3,0,1]), (255,0,0), 2) #z
    cv2.imshow("aaaaaaa", color_img)
    cv2.waitKey(0)    
    print("Done! Press any key to exit")

    return R, T




if __name__ == "__main__":
        
        
    #Example for running intrinsic and extrinsic calibration on provided structured light examples

    #Input data locations
    #for CAMERA calibration:
    baseDir = './data/calib' #data directory
    calName = '12_15_calib' #calibration sourse (also a dir in data)    
    ambient = 'normal_lighting'
    image_ext = 'JPG' #file extension for images




    #Extrinsic calibration parameters
    dW1 = (8, 8) #window size for finding checkerboard corners
    checkerboard = (6, 8) #number of internal corners on checkerboard
    size_of_square = 0.0235 #in cm
    objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[0,:,:2] = (size_of_square*np.mgrid[0:checkerboard[0], 0:checkerboard[1]]).T.reshape(-1, 2)    

    images = glob.glob(os.path.join(baseDir, ambient, "*"+image_ext))

    extr_img = images[0]

    cam_intr_stuff = np.load(os.path.join(baseDir, calName, "cam_intrinsic_calib.npz"))
    cam_mtx = cam_intr_stuff["mtx"]
    cam_dist = cam_intr_stuff["dist"]  


    R,T = get_extrinsic(objp, checkerboard, dW1, cam_mtx, cam_dist, extr_img)


    np.savez(os.path.join(baseDir, calName, "cam_EXTRINSIC_calib.npz"), R=R, T=T)   