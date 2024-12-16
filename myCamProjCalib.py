from cp_hw6 import computeIntrinsic, computeExtrinsic, pixel2ray
import os
import glob
import numpy as np
import cv2
from triangulate import getIMGPaths, getImageStack, classify_imgstack_codes, decode_gray, read_img, get_per_pixel_threshold
import matplotlib.pyplot as plt



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


if __name__ == "__main__":
    #Example for running intrinsic and extrinsic calibration on provided structured light examples

    #Input data locations
    #for CAMERA calibration:
    baseDir = './data/calib' #data directory
    calName = '12_15_calib' #calibration sourse (also a dir in data)
    projName = '12_15_projgray'

    def get_proj_name(i):
        num = i+1
        return f'12_15_{num}_projgray'
    
    ambient = 'normal_lighting'
    
    image_ext = 'JPG' #file extension for images
    skip_cam_intrinsic = True
    skip_cam_extrinsic = True
    load_proj_decoded = True
    
    useLowRes = False #enable lowres for debugging

    
    offsets = np.load("offsets.npy")
    captures = 5



    
    w = 2048
    h = 2048


    def get_slice(img, start_y, start_x, h=2048, w=2048):
        return img[start_y: start_y+h, start_x:start_x+w, :]



    #Extrinsic calibration parameters
    dW1 = (8, 8) #window size for finding checkerboard corners
    checkerboard = (6, 8) #number of internal corners on checkerboard
    size_of_square = 0.0235 #in cm

    normal_lighting_imgs = glob.glob(os.path.join(baseDir, ambient, "*"+image_ext))



    #Part 1: Intrinsic Calibration
    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################
    images = glob.glob(os.path.join(baseDir, calName, "*"+image_ext))

    
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

##############################################################################################################
##############################################################################################################
#############################################################################################################
    

    print("INTRINSIC PARAMETERS:")
    print("Camera matrix: \n")
    print(mtx)
    print("Distortion: \n")
    print(dist) 

    if(not skip_cam_extrinsic):

        corners2 = get_chessboard_imgpts(cv2.imread(images[-1]), checkerboard, dW1)

        ret, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
        cam_R = cv2.Rodrigues(rvec)[0]
        cam_T = tvec.ravel()

        print("EXTRINSIC PARAMETERS:")
        print("Camera matrix: \n")
        print(cam_R)
        print("Translation: \n")
        print(cam_T) 

        color_img = cv2.imread(images[-1])
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


    proj_objps_list = []
    proj_imgpts_list = []

    cam_objps_list = []
    cam_imgpts_list = []    




    for dir_index in range(captures):
        dir_name = get_proj_name(dir_index)
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

            #############GET HOMOGRAPHY#########################
            print("finding homography...")
            for dx in range(-patch_half, patch_half):
                for dy in range(-patch_half, patch_half):
                    x = c_x + dx #column we are seeking to MATCH
                    y = c_y + dy #row value for decoded
                    
                    search_proj_col = np.where(decoded[y] - x < 350)

                    if(len(search_proj_col[0]) == 0):
                        print("1")
                        continue
                    else:
                        proj_col = decoded[y, search_proj_col[0][0]]

                    src_pts.append((x,y)) #from CAMERA
                    dst_pts.append(np.array([x, proj_col])) #where projector found it
            
            if(len(src_pts) == 0 and len(dst_pts) == 0):
                print("2")
                continue          
            #after finding enough points near corner region...
            #############GET HOMOGRAPHY#########################
            Hcam_to_proj, inliers = cv2.findHomography(np.array(src_pts), np.array(dst_pts),cv2.RANSAC, 5.0)
            if(np.any(Hcam_to_proj) is None):
                #print("could not produce homography")
                print("3")
                continue
            print(f"homograpy for corner{corner_index} is \n", Hcam_to_proj)
            homo_proj_pt = Hcam_to_proj @ np.array([corner_x,corner_y, 1])
            proj_pt = homo_proj_pt[0:2]/homo_proj_pt[2]

            proj_objps.append(objpt)
            proj_imgpoints.append([proj_pt]) #the points that the projector found in image 

            cam_2objps.append(objpt)
            cam_2imgpoints.append([np.array(corner_x, corner_y)])               
            
            #only add camera point if projector was found!!!!
        
        print("found points:", len(proj_objps))
        
        proj_objps_list.append(np.float32(proj_objps))
        proj_imgpts_list.append(np.float32(proj_imgpoints))
        
        cam_objps_list.append(objp)
        cam_imgpts_list.append(cam_corners)   
        

    
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


    ret, cam_int, cam_dist, proj_int, proj_dist, cam_proj_rmat, cam_proj_tvec, E, F = cv2.stereoCalibrate(
        proj_objps_list, cam_2imgpoints, proj_imgpts_list, mtx, dist, proj_mtx, proj_dist, None)  


    if ret == True:
        print("STEREO CALIBRATED PARAMETERS!!!")

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

        print("CAM R, T:")
        print("R \n")
        print(cam_proj_rmat)
        print("T \n")
        print(cam_proj_tvec)   

        print("Essential matrix: \n")
        print(E)
        print("Fundamental matrix: \n")
        print(F)      
    else:
        print("stereo calibrate failed")



     