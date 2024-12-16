import skimage.io as ski_io
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from cp_hw6 import pixel2ray, set_axes_equal
import re, os
import cv2
from tqdm import tqdm
import pickle
from util import decode, getImageStack, classify_imgstack_codes, get_per_pixel_threshold, read_img
import glob


def get_center_slice(img, h=2048, w=2048):
    img_shape = img.shape
    center_y = img_shape[0]//2
    center_x = img_shape[1]//2
    return img[center_y-h//2:center_y+h//2, center_x-w//2:center_x+w//2, :]

def get_slice(img, start_y, start_x, h=2048, w=2048):
    return img[start_y: start_y+h, start_x:start_x+w, :]



if __name__ == "__main__":
    
    
    baseDir = './data/calib' #data directory
    calName = '12_15_calib' #calibration sourse (also a dir in data)

    dataDir = './data' #data directory
    patternDir = 'binary' #calibration sourse (also a dir in data)   
    image_ext="JPG" 
    load_proj_decoded = True

    h = 2048
    w = 2048

    #bird offset 1075, 2297]


    
    ###############################################################
    ###############################################################
    ###############################################################
    ###############################################################

    stereo_stuff = np.load(os.path.join(baseDir, calName, "stereo.npz"))
    cam_int = stereo_stuff['cam_int']
    cam_dist = stereo_stuff['cam_dist']

    proj_int = stereo_stuff['proj_int']
    proj_dist = stereo_stuff['proj_dist']

    stereoR = stereo_stuff['stereoR']
    stereoT = stereo_stuff['stereoT']

    E = stereo_stuff['E']
    F = stereo_stuff['F']


    IMG_PATHS = glob.glob(os.path.join(dataDir, patternDir,"*"+image_ext))
    print("IMG paths", IMG_PATHS)

    if(not load_proj_decoded):
        proj_gray_patterns_before = getImageStack(IMG_PATHS)
        proj_gray_patterns = np.zeros((h, w, 3, len(IMG_PATHS)))
        
        for idx in range(len(IMG_PATHS)):
            proj_gray_patterns[:,:,:,idx] = get_slice(proj_gray_patterns_before[:,:,:,idx], 1075, 2297)

        proj_codes = classify_imgstack_codes(proj_gray_patterns, threshold=get_per_pixel_threshold(proj_gray_patterns))
        decoded = decode(patternDir, proj_codes)
        np.save(f"{patternDir}_decoded.npy", decoded)
    else:
        decoded = np.load(f"{patternDir}_decoded.npy")





    


    #SHOW PROJECTOR CORRESPONDENCES!!!!
    fig, ax = plt.subplots(1, 2, figsize=((15,10)))
    ax[0].imshow(get_slice(read_img(IMG_PATHS[-1]), 1075, 2297))
    ax[0].set_title("final img")
    ax[1].imshow(decoded, cmap='jet')
    ax[1].set_title("projector decoded")
    plt.show() 



    ###TRIANGULATE VIA STEREO

    x = np.arange(2048)
    y = np.arange(2048)
    xv, yv = np.meshgrid(x, y)
    pts = np.stack((xv.ravel(), yv.ravel()), axis=-1).astype(np.float32)

    print("pts.shape is", pts.shape)

    #points: nx2 np.float32 array



    proj_undist_points = cv2.undistortPoints(pts, proj_int, proj_dist)    
    cam_undist_points = cv2.undistortPoints(pts, cam_int, cam_dist)   

    perspective_cam = np.array([[1.0,0,0,0],
                                [0,1.0,0,0],
                                [0,0,1.0,0]])
    
    perspective_proj = np.hstack((stereoR, stereoT))

    triag = cv2.triangulatePoints(perspective_cam, 
                                  perspective_proj, 
                                  cam_undist_points.reshape(2,-1), 
                                  proj_undist_points.reshape(2,-1))
    
    triag = triag.T    
    print("shape of triag is", triag.shape)

    pointCloud = (triag[:,:3]/triag[:,3:])

    pointCloud = pointCloud.reshape((2048,2048,3))

    print("shape of pointCloud is", pointCloud.shape)
    
    color = get_slice(read_img(IMG_PATHS[-1]), 1075, 2297)
    print("shape of colors", color.shape)
    fig2 = plt.figure("Projected camera view")
    ax = fig2.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=270, azim=-90, roll=180)
    C = np.array([0,0,0])  
    ax.scatter(C[0], C[1], C[2], s=10, marker="s")    
    ax.scatter(pointCloud[:,0],
               pointCloud[:,1],
               pointCloud[:,2])
    set_axes_equal(ax)  
    
    plt.show()        










    
