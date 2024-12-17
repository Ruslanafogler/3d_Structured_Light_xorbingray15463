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



def decode(pattern_type, img_codes):
    if(pattern_type.lower() == 'binary'):
        return decode_binary(img_codes)
    elif(pattern_type.lower() == 'gray'):
        return decode_gray(img_codes)
    elif(pattern_type.lower() == 'xor'):
        return decode_xor(img_codes)
    else:
        print("unknown input. double check your spelling.")



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



def write_image(path, data, is_float=True):
    if(is_float):
        data = np.clip(data, 0.0, 1.0)
        data = data*255
    saved = data.astype(np.uint8)
    ski_io.imsave(path, saved)

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
