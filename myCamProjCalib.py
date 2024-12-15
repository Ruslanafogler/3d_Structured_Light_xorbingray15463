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
    calName = '12_15_calib' #calibration sourse (also a dir in data)
    projName = '12_15_projgray'
    image_ext = 'JPG' #file extension for images
    skip_cam_intrinsic = True
    skip_cam_extrinsic = False
    load_proj_decoded = True
    
    useLowRes = False #enable lowres for debugging

    extr_img = "./data/calib/normal00013.JPG"

    
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

    proj_images = glob.glob(os.path.join(baseDir, projName, "*"+image_ext))


    print("proj Path is: ", os.path.join(baseDir, projName, "*"+image_ext))
    print("images paths are: \n", proj_images)     

    #Part 1: Intrinsic Calibration
    images = glob.glob(os.path.join(baseDir, calName, "*"+image_ext))
    '''
    last image of images can be used to get homogrpahy for proj_images
    '''

    
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



    """
    Performing camera calibration by
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the
    detected corners (imgpoints)
        ret:
        mtx: camera matrix
        dist: distortion coefficients

    """

    objpoints = []
    imgpoints = []
    img = cv2.imread(extr_img)
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


        img = cv2.drawChessboardCorners(img, checkerboard, corners2, ret)
        resized_image = cv2.resize(img, (800, 600))
        # cv2.imshow('img',resized_image)
        # cv2.setWindowTitle('img', "extr showing") 
        # cv2.waitKey(0)
         
    else:
        print("could not find extrinsic chessboard")
        #img = cv2.drawChessboardCorners(img, checkerboard, corners2, ret)
        resized_image = cv2.resize(img, (800, 600))
        cv2.imshow('img',resized_image)
        cv2.setWindowTitle('img', "extr showing") 
        cv2.waitKey(0)        
    


    #corners2 and these points should be for the FINAL image (upright board)
    

    print("INTRINSIC PARAMETERS:")
    print("Camera matrix: \n")
    print(mtx)
    print("Distortion: \n")
    print(dist) 

    if(not skip_cam_extrinsic):

        ret, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
        cam_R = cv2.Rodrigues(rvec)[0]
        cam_T = tvec.ravel()

        print("EXTRINSIC PARAMETERS:")
        print("Camera matrix: \n")
        print(cam_R)
        print("Translation: \n")
        print(cam_T) 

        color_img = cv2.imread(extr_img)
        color_img = get_slice(color_img)
        
        axis = np.float32([[0,0,0], [1,0,0], [0,1,0], [0,0,1]]).reshape(-1,3)
        axis_img = cv2.projectPoints(axis, rvec, tvec, mtx, dist)[0]
        #print("projected points: ", axis_img)

        axis_img = axis_img.astype(int)

        np.savez(os.path.join(baseDir, calName, "cam_extrinsic_calib.npz"), R=cam_R, T=cam_T)
        #cv2.namedWindow('Result')
        cv2.putText(color_img, 'X', (axis_img[1,0,0], axis_img[1,0,1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255))
        cv2.line(color_img, (axis_img[0,0,0], axis_img[0,0,1]), (axis_img[1,0,0], axis_img[1,0,1]), (0,0,255), 2) #x
        cv2.putText(color_img, 'Y', (axis_img[2,0,0], axis_img[2,0,1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))
        cv2.line(color_img, (axis_img[0,0,0], axis_img[0,0,1]), (axis_img[2,0,0], axis_img[2,0,1]), (0,255,0), 2) #y
        cv2.putText(color_img, 'Z', (axis_img[3,0,0], axis_img[3,0,1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0))
        cv2.line(color_img, (axis_img[0,0,0], axis_img[0,0,1]), (axis_img[3,0,0], axis_img[3,0,1]), (255,0,0), 2) #z
        #cv2.imshow("aaaaaaa", color_img)
        #cv2.waitKey(0)    
        print("Done! Press any key to exit")
    else:
        print("SKIPPING extrinsc routine...")
        extr_stuff = np.load(os.path.join(baseDir, calName, "cam_extrinsic_calib.npz"))
        cam_R = extr_stuff["R"]
        cam_T = extr_stuff["T"]     




    ####PROJECTOR CALIBRATION
    #use white image
    if(not load_proj_decoded):
        proj_gray_patterns_before = getImageStack(proj_images)

        proj_gray_patterns = np.zeros((h, w, 3, len(proj_images)))

        print("shape of proj_gray_patterns")

        

        for idx in range(len(proj_images)):
            proj_gray_patterns[:,:,:,idx] = get_slice(proj_gray_patterns_before[:,:,:,idx])



        # fig, ax = plt.subplots()
        # first_img = read_img(proj_gray_patterns[0])
        # ax.imshow(first_img)

        # print("pick two points, first for black threshold, then for white threshold")

        # points = []

        # def onclick(event):
        #     if len(points) < 2:
        #         points.append((event.xdata, event.ydata))
        #         print(f"Selected point: ({event.xdata}, {event.ydata})")
        #     if len(points) == 2:
        #         plt.close()
    

        # proj_gray_patterns = getIMGPaths("./data/calib/proj", ".JPG")
        
        proj_codes = classify_imgstack_codes(proj_gray_patterns, threshold=get_per_pixel_threshold(proj_gray_patterns))
        decoded = decode_gray(proj_codes)
        np.save("proj_gray_decoded.npy", decoded)
    else:
        decoded = np.load("proj_gray_decoded.npy")


    #SHOW PROJECTOR CORRESPONDENCES!!!!
    fig, ax = plt.subplots(1, 2, figsize=((15,10)))
    ax[0].imshow(get_slice(read_img(extr_img)))
    ax[0].set_title("final img")
    ax[1].imshow(decoded, cmap='jet')
    ax[1].set_title("projector decoded")
    # plt.show()

    first_proj_img = read_img(proj_images[0])
    patch_half = np.ceil(first_proj_img.shape[1]/180).astype(np.int8)  #horz legth/180 <-- this is a neighboorhood of a chess corner

    proj_objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    proj_imgpoints = []


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

    # print("corners 2 is", corners2)
    # print("patch_half is", patch_half)
    
    objp = np.squeeze(objp, axis=0)


    i = 0
    for corner, objpt in zip(corners2, objp):
        # print(f"\n corner {i} is", corner)
        # print(f"\n objp {i} is", objp)
        #chessboard corner
        top_left_x = corner[0][0]
        top_left_y = corner[0][1]
        c_x = int(round(top_left_x)) 
        c_y = int(round(top_left_y)) 
        src_pts = []
        dst_pts = []


        # Display the image
        # cv2.rectangle(img, (c_x-patch_half, c_y-patch_half), (c_x+patch_half, c_y+patch_half), (0, 255, 0), 2)
        # cv2.imshow("Image with Rectangle", img)
        # cv2.waitKey(0)        

        for dx in range(-patch_half, patch_half):
            for dy in range(-patch_half, patch_half):
                x = c_x + dx #column we are seeking to MATCH
                y = c_y + dy #row value for decoded
                
                search_proj_col = np.where(decoded[y] - x < 10)

                if(len(search_proj_col[0]) == 0):
                    continue
                else:
                    proj_col = decoded[y, search_proj_col[0][0]]

                src_pts.append((x,y)) #from CAMERA
                dst_pts.append(np.array([x, proj_col])) #where projector found it

        
        # print("src points are", src_pts)
        # print("dst points are", dst_pts)


        if(len(src_pts) == 0 and len(dst_pts) == 0):
            print(f"couldnt produce homography for corner {i}")
            continue
        
        #after finding enough points near corner region...
        Hcam_to_proj, inliers = cv2.findHomography(np.array(src_pts), np.array(dst_pts),cv2.RANSAC, 5.0)
        print(f"homograpy for corner{i} is \n", Hcam_to_proj)
        if(np.any(Hcam_to_proj) is None):
            print("could not produce homography")
            continue
        
        homo_proj_pt = Hcam_to_proj @ np.array([top_left_x,top_left_y, 1])
        proj_pt = homo_proj_pt[0:2]/homo_proj_pt[2]

        print("shape of objpt is", objpt.shape)

        proj_objpoints.append(objpt)
        proj_imgpoints.append(proj_pt) #the points that the projector found in image 

    
    print(f"{len(proj_objpoints)}obj points are", proj_objpoints)
    print(f"{len(proj_imgpoints)}img points are", proj_imgpoints)

    
    ret, proj_mtx, proj_dist, proj_rvecs, proj_tvecs = cv2.calibrateCamera(proj_objpoints, 
                                                                           proj_imgpoints, 
                                                                           ((h,w)), 
                                                                           None, 
                                                                           None) 


    np.savez(os.path.join(baseDir, calName, "my_intrinsic_calib.npz"), mtx=mtx, dist=dist)

     