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

#points: nx2 np.float32 array
#mtx: camera matrix
#dist: distortion values
#rays: nx1x3 np.float32 array
def pixel2ray(points, mtx, dist):
    #given x, K, dist
    undist_points = cv2.undistortPoints(points, mtx, dist)
    #from x = PX, this is getting X (the 3d points)
    rays = cv2.convertPointsToHomogeneous(undist_points)
    norm = np.sum(rays**2, axis = -1)**.5
    rays = rays/norm.reshape(-1, 1, 1)
    return rays



def get_ray_plane_intersection(ray, ray_pt, plane_pt, normal):
    normal_dot_ray = np.dot(normal, ray)
    if(normal_dot_ray < 1e-6):
        print("uhoh, ray/plane are parallel")
        return np.array([None] * 3)
    
    t = np.dot(normal, plane_pt-ray_pt)/normal_dot_ray
    return ray*t + ray_pt


def intersect_with_plane(pixel, cam_int, cam_dist, proj_int, proj_dist, stereoR, stereoT):
    
    #from C = (0,0,0)
    cam_center = np.zeros((3,1))
    cam_rays = np.squeeze(pixel2ray(pixel, cam_int, cam_dist))
    #to get SL light plane...
    proj_center = stereoR @ (cam_center-stereoT)
    #do I need to convert this to camera space with R, T?
    proj_rays = np.squeeze(pixel2ray(pixel, proj_int, proj_dist))
    proj_rays = stereoR @ (proj_rays) #questioning this line
    SL_plane_normal = np.cross(proj_rays, np.array([0,0,1]))
    proj_rays = np.squeeze(proj_rays)


    # # Camera positions
    # camera1_pos = np.array([0, 0, 0])  # Assuming camera 1 as origin
    # camera2_pos = stereoR @ (camera1_pos-stereoT)
    # # Plot the cameras
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(camera1_pos[0], camera1_pos[1], camera1_pos[2], color='blue', label='Camera')
    # ax.quiver(camera1_pos[0], camera1_pos[1], camera1_pos[2],
    #           cam_rays[0], cam_rays[1], cam_rays[2], 
    #           color='blue', label='Camera Rays')
    # ax.scatter(camera2_pos[0], camera2_pos[1], camera2_pos[2], color='red', label='Projector')
    # ax.quiver(camera2_pos[0], camera2_pos[1], camera2_pos[2], 
    #           proj_rays[0], proj_rays[1], proj_rays[2], 
    #           color='red', label='Projector Rays')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.legend()
    # plt.show()


    intersection = get_ray_plane_intersection(cam_rays, np.squeeze(cam_center), np.squeeze(proj_center), SL_plane_normal)
    return intersection

    
    




    




if __name__ == "__main__":
    
    
    baseDir = './data/calib' #data directory
    calName = '12_15_calib' #calibration sourse (also a dir in data)

    dataDir = './data' #data directory
    patternDir = 'gray' #calibration sourse (also a dir in data)   
    image_ext="JPG" 
    load_proj_decoded = True

    h = 2048
    w = 2048

    offsety = 1075
    offsetx = 2297
    #form a 2048x2048 bounding box around bird following this offset 

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
            proj_gray_patterns[:,:,:,idx] = get_slice(proj_gray_patterns_before[:,:,:,idx], offsety, offsetx)

        proj_codes = classify_imgstack_codes(proj_gray_patterns, threshold=get_per_pixel_threshold(proj_gray_patterns))
        decoded = decode(patternDir, proj_codes)
        np.save(f"{patternDir}_decoded.npy", decoded)
    else:
        decoded = np.load(f"{patternDir}_decoded.npy")





    


    #SHOW PROJECTOR CORRESPONDENCES!!!!
    fig, ax = plt.subplots(1, 2, figsize=((15,10)))
    ax[0].imshow(get_slice(read_img(IMG_PATHS[-1]), offsety, offsetx))
    ax[0].set_title("final img")
    ax[1].imshow(decoded, cmap='jet')
    ax[1].set_title("projector decoded")
    #plt.show() 







    n = 20 #downsample by


    x = np.arange(0, 2048, n)
    y = np.arange(0, 2048, n)
    xv, yv = np.meshgrid(x, y)
    pts = np.stack((xv.ravel(), yv.ravel()), axis=-1).astype(np.float32)

    color = get_slice(read_img(IMG_PATHS[-1]), offsety, offsetx)
    print("shape of colors", color.shape)

    ###TRIANGULATE VIA STEREO

    print("pts.shape is", pts.shape)
    

    ##############################################
    #OPENCV WAY
    ###############################################
    # proj_undist_points = cv2.undistortPoints(pts, proj_int, proj_dist)    
    # cam_undist_points = cv2.undistortPoints(pts, cam_int, cam_dist)   

    # perspective_cam = np.array([[1.0,0,0,0],
    #                             [0,1.0,0,0],
    #                             [0,0,1.0,0]])
    
    # perspective_proj = np.hstack((stereoR, stereoT))

    # triag = cv2.triangulatePoints(perspective_cam, 
    #                               perspective_proj, 
    #                               cam_undist_points.reshape(2,-1), 
    #                               proj_undist_points.reshape(2,-1))
    # triag = triag.T    
    # print("shape of triag is", triag.shape)

    # pointCloud = (triag[:,:3]/triag[:,3:])
    # pointCloud = pointCloud.reshape((x.shape[0],y.shape[0],3))

    # print("shape of pointCloud is", pointCloud.shape)
    ##################################################################

    pointCloud = np.zeros((y.shape[0], x.shape[0], 3))
    
    
    for pt in pts:
        X = intersect_with_plane(pt, cam_int, cam_dist, proj_int, proj_dist, stereoR, stereoT)
        if(X.any() == None): continue
        print("shape of X is", X.shape)
        if(X[2] < 0 and np.abs(X[2]) < 10):
            print("depth at pixel", pt//n, "is", X[2])
            pixel = pt.astype(np.uint8)
            pointCloud[pixel[1]//n, pixel[0]//n] = X[2]


                

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










    
