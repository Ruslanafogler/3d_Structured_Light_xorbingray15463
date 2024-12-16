import skimage.io as ski_io
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from cp_hw6 import pixel2ray, set_axes_equal
import re, os
import cv2
from tqdm import tqdm
import pickle


def get_I_lumn(I, has_4_channels=False):
    if(has_4_channels):
        R = I[:,:,0,:]
        G = I[:,:,1,:]
        B = I[:,:,2,:]        
    else:
        R = I[:,:,0]
        G = I[:,:,1]
        B = I[:,:,2]
    return 0.2126*R + 0.7152*G + 0.0722*B  

def get_per_pixel_threshold(img_stack):
    #y,x,(rgb), # of images
    gray_img_stack = get_I_lumn(img_stack, has_4_channels=True)
    return np.mean(gray_img_stack, axis=2)


def classify_imgstack_codes(img_stack, threshold, test_pattern=False):
    if(not test_pattern):
        gray_img_stack = get_I_lumn(img_stack, has_4_channels=True)
    else:
        gray_img_stack = img_stack #(for confirming with binary patterns)
    r,c, num_imgs = gray_img_stack.shape
    img_codes = np.zeros_like(gray_img_stack)
    for i in range(num_imgs):
        img = gray_img_stack[:,:,i]
        new_img = np.where(img > threshold, 1.0, 0.0)
        img_codes[:,:,i] = new_img
    return img_codes


def decode_binary(img_codes):
    y,x,num_codes = img_codes.shape
    print("shape of img_codes is", y, x, num_codes)
    pows_of_two = 1 << np.arange(0, num_codes)
    pows_of_two = pows_of_two[::-1]
    decoded = np.sum(img_codes * pows_of_two, axis=2)
    return decoded


def decode_gray(img_codes):
    y,x,num_codes = img_codes.shape
    print("shape of img_codes is", y, x, num_codes)
    pows_of_two = 1 << np.arange(0, num_codes)
    pows_of_two = pows_of_two[::-1]
    #recusive conversion for gray2bin
    print("type of img_codes is", img_codes.dtype)
    
    img_codes = img_codes.astype(int)

    bin_codes = gray2bin(img_codes, num_codes)
    
    
    # for i in range(1, num_codes):
    #     img_codes[:,:,i] = np.bitwise_xor(img_codes[:,:,i], img_codes[:,:,i-1])

    decoded = np.sum(bin_codes * pows_of_two, axis=2)
    return decoded


def gray2bin(gray_codes, num_imgs):
    for i in range(1, num_imgs):
        gray_codes[:,:,i] = np.bitwise_xor(gray_codes[:,:,i], gray_codes[:,:,i-1]) 
    return gray_codes


def decode_xor(img_codes, final_img_ind=-1):
    y,x,num_codes = img_codes.shape
    print("shape of img_codes is", y, x, num_codes)
    pows_of_two = 1 << np.arange(0, num_codes)
    pows_of_two = pows_of_two[::-1]
    #recusive conversion for gray2bin
    print("type of img_codes is", img_codes.dtype)
    img_codes = img_codes.astype(int)

    #xor to gray
    # default to xor-4 (final plane is LAST img), xor2 is second-to-last
    last_img_in_stack = img_codes[:,:,final_img_ind]
    imgs_gray = np.empty((y, x, num_codes), dtype=np.uint8)
    for i in range(num_codes):
        imgs_gray[:,:,i] = np.bitwise_xor(img_codes[:,:,i], last_img_in_stack)
    imgs_gray[:,:,final_img_ind] = last_img_in_stack.copy()    

    bin_code = gray2bin(imgs_gray, num_codes) 
    decoded = np.sum(bin_code * pows_of_two, axis=2)
    return decoded


####taken from cp_hw6 code
#points: nx2 np.float32 array
#intr_mxtx: camera/projector intrinsic matrix
#dist: distortion values
#rays: nx1x3 np.float32 array
def pixel2ray(points, intr_mxtx, dist):
    #given x, K, dist
    undist_points = cv2.undistortPoints(points, intr_mxtx, dist)
    #from x = PX, this is getting X (the 3d points)
    rays = cv2.convertPointsToHomogeneous(undist_points)
    #print("rays: ", rays)
    norm = np.sum(rays**2, axis = -1)**.5
    #print("norm: ", norm)
    rays = rays/norm.reshape(-1, 1, 1)
    return np.squeeze(rays)




def get_plane_intersection(camera_3dpt, rays):
    #camera and rays are in respect to PLANE coords!!!
    #rays are 3x2
    intersections = np.zeros_like(rays)
    camera_z_coord = camera_3dpt[2,:]
    rays_z_scalar = rays[2,:]
    t = -camera_z_coord/rays_z_scalar
    intersections = camera_3dpt + rays*t
    return intersections


def int_to_world(camera_3dpt, R, T):
    shape_of_input = camera_3dpt.shape
    assert shape_of_input[0] == 3, f"rows should be 3 long, instead shape was {shape_of_input}"
    transformed = np.dot(R.T,(camera_3dpt - T)) 
    return transformed


def world_to_int(plane_3dpt, R, T):
    shape_of_input = plane_3dpt.shape
    assert shape_of_input[0] == 3, f"rows should be 3 long, instead shape was {shape_of_input}"
    transformed = np.dot(R, plane_3dpt) + T
    return transformed

def get_camera_ray(point, intr_c, dist, R_c, T_c):
    ray_camera = pixel2ray(point, intr_c, dist)
    ray_world = int_to_world(ray_camera, R_c, np.zeros(3,1))
    return ray_world

def center_to_world(R, T):
    return int_to_world(np.zeros((3,1)), R, T)


def get_ray_plane_intersection(ray, ray_pt, plane_pt, normal):
    #Ax + By + Cz = 0
    #[x,y,z] = rt + [x0, y0, z0]
    A = normal[0]
    B = normal[1]
    C = normal[2]
    D = -np.dot(plane_pt, normal)


    #t = (D-np.dot(normal, ray))/(D-np.dot(normal,ray_pt))
    t= np.dot(normal, plane_pt-ray_pt)/np.dot(normal, ray)
    
    return ray*t



def get_cam_proj_corresponence(ray_world, proj_plane_pt, proj_normal):
    intersection = get_ray_plane_intersection(ray_world, proj_plane_pt, proj_normal)
    return intersection[:,2]


def get_proj_planes(img_codes, intr_p, dist, R_p, T_p):
    row_len, col_len = img_codes.shape
    r1 = row_len//3
    r2 = 2*row_len//3
    plane_normals = []
    plane_pts = []
    for col in range(col_len):
        # img_column = img_codes[:,col_index]
        pray_projector = pixel2ray(np.array([[r1, col], [r2, col]]), intr_p, dist)
        pray_world = int_to_world(pray_projector, R_p, np.zeros((3,1)))
        pcenter_world = int_to_world(np.zeros((3,1)), R_p, T_p)
        P1 = pray_world[:, 0]
        P2 = pray_world[:, 1]
        P3 = pcenter_world

        normal = np.cross((P2-P1),(P3-P2))
        unit_normal = normal/np.linalg.norm(normal)
        
        plane_pts.append(P1)
        plane_normals.append(unit_normal)
    return plane_pts, plane_normals


# def get_proj_plane(col, intr_p, R_p, T_p, dist, img_codes):
#     row_len, col_len = img_codes.shape
#     r1 = row_len//3
#     r2 = 2*row_len//3
#     proj_center = camera_to_plane(np.zeros((3,1)), R_p, T_p)
#     plane_normals = []
#     plane_pts = []

#     for col in range(img_codes.shape[1]):
#         # img_column = img_codes[:,col_index]
#         projector_ray = pixel2ray(np.array([[r1, col], [r2, col]]), intr_p, dist)
#         intersections_plane = intersect_plane_z_zero(proj_center, projector_ray)
#         P1 = projector_ray[:, 0]
#         P2 = projector_ray[:, 1]
#         P3 = intersections_plane[:, 0]
#         P4 = intersections_plane[:, 1]
#         normal = np.cross((P2-P1),(P4-P3))
#         unit_normal = normal/np.linalg.norm(normal)
        
#         plane_pts.append(P1)
#         plane_normals.append(unit_normal)
#     return plane_pts, plane_normals


def intersect_proj_plane_camera_ray():
    pass

#code inspired from https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def getIMGPaths(prefix_path, suffix):
    images = []
    #prefix_path = '../data/'
    for f in os.listdir(prefix_path):
        split_tup = os.path.splitext(f);        
        if(split_tup[1] == suffix):
            images.append(prefix_path + '/' + f)
    
    images.sort(key = natural_keys)
    return images;

def read_img(path, downsample=False, n=10):
    img = ski_io.imread(path)
    if(downsample):
        img = img[::10, ::10]
    return img

def getImageStack(IMG_PATHS):
    img_stack = []
    t = 0
    for path in IMG_PATHS:
        I_t = read_img(path, False)
        print(f"range of image is {t}", np.min(I_t), np.max(I_t))
        print(f"shape of image {t} is", I_t.shape)
        img_stack.append(I_t)
        t+=1
    img_stack = np.array(img_stack)
    img_stack = np.moveaxis(img_stack, 0, -1) 
    print("shape of image stack is:", img_stack.shape)
    return img_stack




def main():

    ##DECODING

    #GRAY_PATHS = getIMGPaths("./data/gray/", ".JPG")

    BINARY_PATHS = getIMGPaths("./data/binary", ".JPG")
    BINARY_PATHS = BINARY_PATHS[:-2]
    print("BINARY paths are", BINARY_PATHS)


    #or make a list....
    #gray_imgs_stack = getImageStack(GRAY_PATHS)
    #bin_imgs_stack = getImageStack(BINARY_PATHS)
    
    # gray_patterns = getIMGPaths("./patterns/gray", ".tiff")
    # gray_patterns = getImageStack(gray_patterns)
    # gray_codes = classify_imgstack_codes(gray_patterns, threshold=127.0, test_pattern=True)
    # decoded1 = decode_gray(gray_codes)

    # bin_patterns = getIMGPaths("./patterns/binary", ".tiff")
    # bin_patterns = getImageStack(bin_patterns)
    # bin_codes = classify_imgstack_codes(bin_patterns, threshold=127.0)
    # decoded2 = decode_binary(bin_codes)

    # xor_patterns = getIMGPaths("./patterns/xor", ".tiff")
    # xor_patterns = getImageStack(xor_patterns)
    # xor_codes = classify_imgstack_codes(xor_patterns, threshold=127.0, test_pattern=True)
    # decoded1 = decode_xor(xor_codes)    




    
    patterns = getIMGPaths("./data/calib/12_14_birdxor", ".JPG")
    patterns = getImageStack(patterns)
    codes = classify_imgstack_codes(patterns, threshold=127.0, test_pattern=False)
    decoded1 = decode_binary(codes)

    # plt.imshow(decoded2, cmap='jet')
    # plt.title("binary")
    # plt.show()

    plt.imshow(decoded1, cmap='jet')
    plt.title("xor")
    plt.show()
    return







    return

    ###TRIANGULATION
    cam_intr_stuff = np.load("./data/calib/my_intrinsic_calib.npz")
    #extr_stuff = np.load("./data/my-data/need/prmy_extrinsic_calib.npz")
    
    intr_p = cam_intr_stuff["mtx"]
    dist_p = cam_intr_stuff["dist"]
    rvecs_P = cam_intr_stuff["rvecs"]
    tvecs_P = cam_intr_stuff["tvecs"]
    extr_R_p, _ = cv2.Rodrigues(rvecs_P)
    extr_T_p = tvecs_P.ravel() 


    #projector related matrices
    intr_p = 5
    dist_p = 5
    extr_R_p = 5
    extr_T_p = 5

    #camera related matrices
    intr_c = 5
    dist_c = 10
    extr_R_c = 5
    extr_T_c = 5

    img_codes = np.zeros(5,5)

    SL_MIN_X = 303
    SL_MAX_X = 815

    SL_MIN_Y = 304
    SL_MAX_Y = 655

    #extra todos--> filter out pattern results and make sure that columns
    #have actually matching codes??

    #write image capture code and pipeline (projector time!)
    #add debuggin visualizations 
    #write pipeline for getting the image stack and the rest, ofc


    #triangulating part...
    proj_planes, proj_normals = get_proj_planes(img_codes, intr_p, dist_p, extr_R_p, extr_T_p)
    reconstructed = np.zeros((SL_MAX_Y-SL_MIN_Y, SL_MAX_X-SL_MIN_X))
    for row in range(SL_MIN_Y, SL_MAX_Y):
        for col in range(SL_MIN_X, SL_MAX_X):
            img_ray = get_camera_ray(np.array([col, row]), 
                                     intr_c, dist_c, 
                                     extr_R_c, extr_T_c)
            corrsp = get_cam_proj_corresponence(img_ray, proj_planes[col], proj_normals[col])
            reconstructed[row,col] = corrsp
    






    pass

if __name__ == "__main__":
    main()

