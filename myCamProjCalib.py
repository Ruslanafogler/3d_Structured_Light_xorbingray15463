from cp_hw6 import computeIntrinsic, computeExtrinsic, pixel2ray
import os
import glob
import numpy as np
import cv2
from util import getIMGPaths, getImageStack, classify_imgstack_codes, decode_gray, read_img, get_per_pixel_threshold
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.001)

def get_chessboard_imgpts(img, checkerboard, dW1, show_img=False):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)
    if ret == True:
        corners = cv2.cornerSubPix(gray, corners, dW1, (-1,-1), criteria)
        img = cv2.drawChessboardCorners(img, checkerboard, corners, ret)
    else:
        print("error: checkerboard not found")

    if(show_img):
        resized_image = cv2.resize(img, (800, 600))
        cv2.imshow('img',resized_image)
        cv2.setWindowTitle('img', "chessboard") 
        cv2.waitKey(0)
    cv2.destroyAllWindows()    
    return corners

def get_proj_name(i, prefix, suffix):
    num = i+1
    return f'{prefix}_{num}'

def get_slice(img, start_y, start_x, h=2048, w=2048):
    return img[start_y: start_y+h, start_x:start_x+w, :]

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
    #Example for running intrinsic and extrinsic calibration on provided structured light examples

    #Input data locations
    #for CAMERA calibration:
    baseDir = './data/calib/' #data directory
    calName = '12_16_calib/cropped' #calibration sourse (also a dir in data)
    projName = '12_15_projgray'
    ambient = '12_16_normallighting_projgray'

    proj_prefix = '12_16_projgray'
    proj_suffix = ''
    

    w = 2048
    h = 2048

    captures = 4 #number of directories for projector calibration
    image_ext = 'JPG' #file extension for images
    skip_cam_intrinsic = True
    load_proj_decoded = False

    get_offsets = False
    

    #Extrinsic calibration parameters
    dW1 = (8, 8) #window size for finding checkerboard corners
    checkerboard = (6, 8) #number of internal corners on checkerboard
    size_of_square = 0.0235 #in cm

    normal_lighting_imgs = glob.glob(os.path.join(baseDir, ambient, "*"+image_ext))    
    images = glob.glob(os.path.join(baseDir, calName, "*"+image_ext))

    print("normal lighting paths:", normal_lighting_imgs)
    print("camera intrinsic calib paths:", images)
    
    
    if(get_offsets):
        offsets = []
            # Example usage:
        for i, p in enumerate(normal_lighting_imgs):
            print("i, p", i, p)
            points = select_points(p)
            print("point is", points)
            offsets.append(points)
            x1, y1 = points[0]

        offsets = np.vstack(offsets)
        print("offsets are", offsets)
        np.save("offsets.npy", offsets)    
        
    
    
    offsets = np.load("offsets.npy")




    #Part 1: Intrinsic Calibration
    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################
    

    
    ################################################
    ################################################
    ################################################


    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[0,:,:2] = (size_of_square*np.mgrid[0:checkerboard[0], 0:checkerboard[1]]).T.reshape(-1, 2)
    img_shape = None

    # Extracting path of individual image stored in a given directory
    
    if(not skip_cam_intrinsic):
        print("DOING intrinsc routine...")
        print("Path is: ", os.path.join(baseDir, calName, "*"+image_ext))
        print("images paths are: \n", images)        

        print('Displaying chessboard corners. Press any button to continue to next example')
        for fname in images:
            img = cv2.imread(fname)
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

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, checkerboard, corners2, ret)
            else:
                print("error: checkerboard not found")

            resized_image = cv2.resize(img, (800, 600))
            cv2.imshow('img',resized_image)
            cv2.setWindowTitle('img', fname) 
            cv2.waitKey(0)
        
        cv2.destroyAllWindows()
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None) 
        np.savez(os.path.join(baseDir, calName, "cam_intrinsic_calib.npz"), mtx=mtx, dist=dist)

    else:
            print("SKIPPING intrinsc routine...")
            intr_stuff = np.load(os.path.join(baseDir, calName, "cam_intrinsic_calib.npz"))
            mtx = intr_stuff["mtx"]
            dist = intr_stuff["dist"]            

##############################################################################################################
##############################################################################################################
#############################################################################################################
    

    print("INTRINSIC PARAMETERS:")
    print("Camera matrix: \n")
    print(mtx)
    print("Distortion: \n")
    print(dist) 

    proj_objps_list = []
    proj_imgpts_list = []

    cam_objps_list = []
    cam_imgpts_list = []    


    for dir_index in range(captures):
        dir_name = get_proj_name(dir_index, proj_prefix, proj_suffix)
        print("USING", dir_name)
        proj_images = glob.glob(os.path.join(baseDir, dir_name, "*"+image_ext))
        offset = offsets[dir_index,:]


        cam_ref_path = normal_lighting_imgs[dir_index]
        cam_ref_img = cv2.imread(cam_ref_path)
        cam_ref_img = get_slice(cam_ref_img, offset[0], offset[1])
        cam_corners = get_chessboard_imgpts(cam_ref_img, checkerboard, dW1)


        proj_objps = []
        proj_imgpoints = []

        #for stereo calibrate call at the end...
        cam_2objps = []
        cam_2imgpoints = []        
        ####PROJECTOR CALIBRATION
        #use white image
        if(not load_proj_decoded):
            proj_gray_patterns_before = getImageStack(proj_images)
            proj_gray_patterns = np.zeros((h, w, 3, len(proj_images)))
            

            for idx in range(len(proj_images)):
                proj_gray_patterns[:,:,:,idx] = get_slice(proj_gray_patterns_before[:,:,:,idx], offset[0], offset[1])

            
            proj_codes = classify_imgstack_codes(proj_gray_patterns, threshold=get_per_pixel_threshold(proj_gray_patterns))
            decoded = decode_gray(proj_codes)
            np.save(f"projgray_{dir_index}_decoded.npy", decoded)
        else:
            decoded = np.load(f"projgray_{dir_index}_decoded.npy")


        #SHOW PROJECTOR CORRESPONDENCES!!!!
        fig, ax = plt.subplots(1, 2, figsize=((15,10)))
        ax[0].imshow(cam_ref_img)
        ax[0].set_title("final img")
        ax[1].imshow(decoded, cmap='jet')
        ax[1].set_title("projector decoded")
        plt.show() 

        first_proj_img = read_img(proj_images[0])
        patch_half = np.ceil(2048/180).astype(np.int8)  #horz legth/180 <-- this is a neighboorhood of a chess corner


        # img = cv2.drawChessboardCorners(img, checkerboard, corners2, ret)
        # resized_image = cv2.resize(img, (800, 600))
        # cv2.imshow('img',resized_image)
        # cv2.setWindowTitle('img', "final image of intr calibration, corner mapping") 
        # cv2.waitKey(0)


        # print("camera collected corners are\n", corners2)
        # print("camera IMGPTS are\n", imgpoints)
        # print("objpoints array is", objpoints)
        

        '''
        much code inspired from SLSCalib paper, which describes
        finding local homographies for each camera-detected point 

        procam-calibration code here:
        https://github.com/kamino410/procam-calibration/blob/main/calibrate.py#L156 
        '''

        TEST_PTS = []
        print("STARTING POINT SEARCH!!! for dir", dir_index)
        for corner_index in range(len(cam_corners)):
            objpt = objp[0][corner_index]
            corner_x = cam_corners[corner_index][0][0]
            corner_y = cam_corners[corner_index][0][1]
            
            c_x = int(round(corner_x)) 
            c_y = int(round(corner_y)) 
            src_pts = []
            dst_pts = []

            # Display the image
            # cv2.rectangle(cam_ref_img, (c_x-patch_half, c_y-patch_half), (c_x+patch_half, c_y+patch_half), (0, 255, 0), 2)
            # cv2.imshow("Image with Rectangle", cam_ref_img)
            # cv2.waitKey(0)   
            # 

            print("searching roughly for corner (x,y)", c_x, c_y)     

            #############GET HOMOGRAPHY#########################
            print("finding homography...")
            for dx in range(-patch_half, patch_half):
                for dy in range(-patch_half, patch_half):
                    x = c_x + dx #column we are seeking to MATCH
                    y = c_y + dy #row value for decoded
                    
                    tolerance_search = 10
                    search_proj_col = np.where(np.abs(decoded[y,:] - x) < tolerance_search)
                    if(len(search_proj_col[0]) == 0):
                        print("1")
                        continue
                    else:
                        proj_col = decoded[y, search_proj_col[0][0]]

                    src_pts.append((x,y)) #from CAMERA
                    dst_pts.append(np.array([proj_col, y])) #where projector found it
            
            if(len(src_pts) < 4 and len(dst_pts) < 4):
                print("2")
                continue          

            Hcam_to_proj, inliers = cv2.findHomography(np.array(src_pts), np.array(dst_pts),cv2.RANSAC, 5.0)
            if(np.any(Hcam_to_proj) is None):
                #print("could not produce homography")
                print("3")
                continue
            print(f"homograpy for corner{corner_index} is \n", Hcam_to_proj)
            homo_proj_pt = Hcam_to_proj @ np.array([corner_x,corner_y, 1])
            proj_pt = homo_proj_pt[0:2]/homo_proj_pt[2]

            TEST_PTS.append(proj_pt)

            proj_objps.append(objpt)
            proj_imgpoints.append([proj_pt]) #the points that the projector found in image 

            cam_2objps.append(objpt)
            cam_2imgpoints.append([np.array([corner_x, corner_y])])               
            
            #only add camera point if projector was found!!!!
        

        TEST_PTS = np.vstack(TEST_PTS)
        fig, ax1 = plt.subplots(1, 2, figsize=((15,10)))
        ax1[0].imshow(cam_ref_img)
        ax1[0].set_title("final img")
        ax1[0].scatter(c_x, c_y, color='red', s=50)
        ax1[1].imshow(decoded, cmap='jet')
        ax1[1].scatter(TEST_PTS[:,0], TEST_PTS[:,1], color='red', s=50)
        ax1[1].set_title("projector decoded")
        plt.show()       

        if(len(proj_objps) < 4):
            print("insufficient points for calibration, skipping")
            continue

        proj_objps_list.append(np.float32(proj_objps))
        proj_imgpts_list.append(np.float32(proj_imgpoints))
        
        cam_objps_list.append(np.float32(cam_2objps))
        cam_imgpts_list.append(np.float32(cam_2imgpoints))   
        

    
    cam_ref_path = normal_lighting_imgs[dir_index]
    cam_ref_img = cv2.imread(cam_ref_path)
    gray = cv2.cvtColor(cam_ref_img,cv2.COLOR_BGR2GRAY)
    img_shape = gray.shape[::-1]

    print("some basic checks...")
    print("proj objpts list is", len(proj_objps_list))
    print("proj imgpts list is", len(proj_imgpts_list))
    
    if len(proj_objps_list) == len(proj_imgpts_list) and len(proj_objps_list) > 0:
        ret, proj_mtx, proj_dist, proj_rvecs, proj_tvecs = cv2.calibrateCamera(proj_objps_list, 
                                                                            proj_imgpts_list, 
                                                                            img_shape, 
                                                                            None, 
                                                                            None, 
                                                                            None, 
                                                                            None) 
    else:
        print("need more points...")


    print("INTRINSIC PARAMETERS PROJECTOR:")
    print("Projector matrix: \n")
    print(proj_mtx)
    print("Distortion: \n")
    print(proj_dist) 


    np.savez(os.path.join(baseDir, calName, "projector_intrinsic_calib.npz"), mtx=proj_mtx, dist=proj_dist)

    print("SKIPPING intrinsc routine...")
    intr_stuff = np.load(os.path.join(baseDir, calName, "cam_intrinsic_calib.npz"))
    mtx = intr_stuff["mtx"]
    dist = intr_stuff["dist"]    


    print("some basic checks...")
    print("proj objpts list is", len(proj_objps_list))
    print("proj imgpts list is", len(proj_imgpts_list))
    print("cam imgpts list is", len(cam_imgpts_list))

    ret, cam_int, cam_dist, proj_int, proj_dist, stereoR, stereoT, E, F = cv2.stereoCalibrate(
        proj_objps_list, cam_imgpts_list, proj_imgpts_list, mtx, dist, proj_mtx, proj_dist, None)  


    print("STEREO CALIBRATED PARAMETERS!!!, note that retval is", ret)
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

    np.savez(os.path.join(baseDir, calName, "stereo.npz"), 
             cam_int=cam_int, 
             cam_dist=cam_dist,
             proj_int=proj_int,
             proj_dist=proj_dist,
             stereoR=stereoR,
             stereoT=stereoT,
             E=E,
             F=F)

     