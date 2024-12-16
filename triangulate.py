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
import matplotlib.image as mpimg
import matplotlib.patches as patches

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
        return np.array([None] * 3)
    
    t = np.dot(normal, plane_pt-ray_pt)/normal_dot_ray
    return ray*t + ray_pt


def intersect_with_plane(cam_pixel, proj_pixel, cam_int, cam_dist, proj_int, proj_dist, stereoR, stereoT, show_3d=False):
    
    #from C = (0,0,0)
    cam_center = np.zeros((3,1))
    cam_rays = np.squeeze(pixel2ray(cam_pixel, cam_int, cam_dist))
    #to get SL light plane...
    proj_center = stereoR @ (cam_center - stereoT)
    #do I need to convert this to camera space with R, T?
    proj_rays = np.squeeze(pixel2ray(proj_pixel, proj_int, proj_dist))
    proj_rays = stereoR @ (proj_rays) #questioning this line
    proj_z_up = stereoR @ np.array([0,0,1])
    SL_plane_normal = np.cross(proj_rays, proj_z_up)
    proj_rays = np.squeeze(proj_rays)


    if(show_3d):
        # Camera positions
        camera1_pos = np.array([0, 0, 0])  # Assuming camera 1 as origin
        camera2_pos = stereoR @ (camera1_pos-stereoT)
        # Plot the cameras
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(camera1_pos[0], camera1_pos[1], camera1_pos[2], color='blue', label='Camera')
        ax.quiver(camera1_pos[0], camera1_pos[1], camera1_pos[2],
                cam_rays[0], cam_rays[1], cam_rays[2], 
                color='blue', label='Camera Rays')
        ax.scatter(camera2_pos[0], camera2_pos[1], camera2_pos[2], color='red', label='Projector')
        ax.quiver(camera2_pos[0], camera2_pos[1], camera2_pos[2], 
                proj_rays[0], proj_rays[1], proj_rays[2], 
                color='red', label='Projector Rays')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        plt.show()


    intersection = get_ray_plane_intersection(cam_rays, np.squeeze(cam_center), np.squeeze(proj_center), SL_plane_normal)
    return intersection

    
    




    
    
def select_points(img_path):
    """
    Displays an image and allows the user to select two points.
    Returns the coordinates of the selected points.

    Args:
        img_path (str): Path to the image file.

    Returns:
        tuple: A tuple containing the coordinates of the two selected points.
    """

    img = mpimg.imread(img_path)

    fig, ax = plt.subplots()
    ax.imshow(img)

    points = []

    def onclick(event):
        x, y = int(event.xdata), int(event.ydata)
        points.append(np.array([y, x]))
        rect = patches.Rectangle((x, y), 2048, 2048, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

    return points




if __name__ == "__main__":
    
    
    baseDir = './data/new-calib' #data directory
    calName = '12_16_calib/cropped' #calibration sourse (also a dir in data)

    dataDir = './new_data' #data directory
    pattern = 'xor' #calibration sourse (also a dir in data) 
    patternDir = 'xor_wad' 
    image_ext="JPG" 
    load_proj_decoded = False

    h = 2048
    w = 2048


    IMG_PATHS = glob.glob(os.path.join(dataDir, patternDir, "*"+image_ext))
    
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

    print("STEREO CALIBRATED PARAMETERS!")
    print("CAM INTRINSIC:")
    print("camera matrix: \n")
    print(cam_int)
    print("Distortion: \n")
    print(cam_dist)         

    print("PROJ INTRINSIC:")
    print("Projector matrix: \n")
    print(proj_int)
    print("Distortion: \n")
    print(proj_dist)     

    print("stereo R, T:")
    print("R \n")
    print(stereoR)
    print("T \n")
    print(stereoT)   

    print("Essential matrix: \n")
    print(E)
    print("Fundamental matrix: \n")
    print(F)   





    print("\n\n")
    print("reading from...",os.path.join(dataDir, patternDir, "*"+image_ext) )
    print("IMG PATHS ARE", IMG_PATHS)
    print("Select a point to form top left corner of 2048x2048 bounding box")
    # points = select_points(IMG_PATHS[-1])
    # print("point is", points)
    # offsety, offsetx = points[0]


    ##FOR BIRD
    # offsety = 918
    # offsetx = 2297

    ##FOR AMONG US
    offsety = 869
    offsetx = 2394
 




# Select a point to form top left corner of 2048x2048 bounding box
# point is [array([ 918, 2297])]
# size of final img (should be hx    

    FINAL_IMG = get_slice(read_img(IMG_PATHS[-1]), offsety, offsetx)
    print("size of final img (should be hx2)", FINAL_IMG.shape)
    print("IMG paths", IMG_PATHS)

    if(not load_proj_decoded):
        proj_gray_patterns_before = getImageStack(IMG_PATHS)
        proj_gray_patterns = np.zeros((h, w, 3, len(IMG_PATHS)))
        
        for idx in range(len(IMG_PATHS)):
            proj_gray_patterns[:,:,:,idx] = get_slice(proj_gray_patterns_before[:,:,:,idx], offsety, offsetx)

        proj_codes = classify_imgstack_codes(proj_gray_patterns, threshold=get_per_pixel_threshold(proj_gray_patterns))
        decoded = decode(pattern, proj_codes)
        np.save(f"{pattern}_decoded.npy", decoded)
    else:
        decoded = np.load(f"{pattern}_decoded.npy")





    


    #SHOW PROJECTOR CORRESPONDENCES!!!!
    fig, ax = plt.subplots(1, 2, figsize=((15,10)))
    ax[0].imshow(FINAL_IMG)
    ax[0].set_title("final img")
    ax[1].imshow(decoded, cmap='jet')
    ax[1].set_title("projector decoded")
    plt.show() 



    n = 5 #downsample by
    i = 0
    
    show_correspondence = True

    cam_pts = []
    proj_pts = []
    color_pts = []
    for x in range(0, w, n):
        for y in range(0, h, n):
            tolerance_search = 8
            search_proj_col = np.where(np.abs(decoded[y,:] - x) < tolerance_search)

            i+=1
            if(len(search_proj_col[0]) == 0):
                #print("1")
                continue
            else:
                proj_col = decoded[y, search_proj_col[0][0]]

            color_pts.append(FINAL_IMG[y,x,:])
            cam_pts.append(np.array([x,y])) #from CAMERA
            proj_pts.append(np.array([proj_col, y])) #where projector found it
            
            if(i%1000000 == 0 and show_correspondence):
                fig, ax1 = plt.subplots(1, 2, figsize=((15,10)))
                ax1[0].imshow(FINAL_IMG)
                ax1[0].set_title("final img")
                ax1[0].scatter(x, y, color='red', s=50)
                
                ax1[1].imshow(decoded, cmap='jet')
                ax1[1].scatter(proj_col, y, color='red', s=50)
                ax1[1].set_title("projector decoded")
                plt.show()             
            

    proj_pts = np.vstack(proj_pts).astype(np.float32)   
    cam_pts = np.vstack(cam_pts).astype(np.float32)
    color_pts = np.vstack(color_pts).astype(np.float32)
    
    if(show_correspondence):
        fig, ax1 = plt.subplots(1, 2, figsize=((15,10)))
        ax1[0].imshow(FINAL_IMG)
        ax1[0].set_title("final img")
        ax1[0].scatter(cam_pts[:,0], cam_pts[:,1], color='red', s=50)
        
        ax1[1].imshow(decoded, cmap='jet')
        ax1[1].scatter(proj_pts[:,0], proj_pts[:,1], color='red', s=50)
        ax1[1].set_title("projector decoded")
        plt.show()   


    print("shape of proj, cam", proj_pts.shape, cam_pts.shape)
    

    ##############################################
    #OPENCV WAY
    ###############################################
    proj_undist_points = cv2.undistortPoints(proj_pts, proj_int, proj_dist)    
    cam_undist_points = cv2.undistortPoints(cam_pts, cam_int, cam_dist)   

    perspective_cam = np.array([[1.0, 0,   0,   0],
                                [0,   1.0, 0,   0],
                                [0,   0,   1.0, 0]])
    
    perspective_proj = np.hstack((stereoR, stereoT))

    triag = cv2.triangulatePoints(perspective_cam,
                                  perspective_proj, 
                                  cam_undist_points, 
                                  proj_undist_points)
    triag = triag.T    

    pointCloud = (triag[:,:3]/triag[:,3:])
    no_outliers_mask = (np.abs(pointCloud[:,2])) < 10
    pointCloud = pointCloud[no_outliers_mask]   
    pointCloud_color = color_pts[no_outliers_mask] 

    print("pointCloud is", pointCloud)
    print("pointCloud is", pointCloud.shape)
    print("color_pts shape is", color_pts.shape)
    ##################################################################
    if(color_pts.shape[0] > pointCloud.shape[0]):
        print("cropping color pts, it is too large somehow")
        color_pts = color_pts[:pointCloud.shape[0], :]
        print("color_pts shape is", color_pts.shape)

    ##############################################
    #RAY-PLANE INTERSECTION WAY
    ###############################################    
    # reconstructed = []
    # colors_reconstructed = []
    # for cam_pt, proj_pt in zip(cam_pts, proj_pts):
    #     X = intersect_with_plane(cam_pt, proj_pt, cam_int, 
    #                              cam_dist, proj_int, 
    #                              proj_dist, stereoR, stereoT,
    #                              show_3d=False)
    #     if(X.any() == None): continue
    #     reconstructed.append(X)
    #     colors_reconstructed.append(FINAL_IMG[cam_pt[1].astype(np.uint8), cam_pt[0].astype(np.uint8), :])
    
    # reconstructed = np.vstack(reconstructed)
    # colors_reconstructed = np.vstack(colors_reconstructed)

    # no_outliers_mask = (np.abs(reconstructed[:,2])) < 10
    # reconstructed = reconstructed[no_outliers_mask]    
    # colors_reconstructed = colors_reconstructed[no_outliers_mask]  

    # print("shape is", reconstructed.shape)


                

    fig2 = plt.figure("Projected camera view")
    ax = fig2.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    C = np.array([0,0,0])  
    ax.scatter(C[0], C[1], C[2], s=10, marker="s")    
    ax.scatter(pointCloud[:,0],
               pointCloud[:,1],
               pointCloud[:,2],
               c=color_pts/255.0)
    ax.view_init(elev=-79, azim=-90, roll=0)
    ax.set_zlim(0.4, 1.4)
    set_axes_equal(ax) 

    # ax1 = fig2.add_subplot(122, projection='3d')
    # ax1.set_xlabel('X')
    # ax1.set_ylabel('Y')
    # ax1.set_zlabel('Z')
    # ax1.view_init(elev=270, azim=-90, roll=180)
    # C = np.array([0,0,0])  
    # ax1.scatter(C[0], C[1], C[2], s=10, marker="s")    
    # ax1.scatter(reconstructed[:,0],
    #            reconstructed[:,1],
    #            reconstructed[:,2],
    #            c=colors_reconstructed/255.0)
    # set_axes_equal(ax)      
    
    plt.show()        










    
