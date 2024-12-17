## IMPORTANT FILES:
# myCamProjCalibration
contains code for calibrating intrinsic parameters of camera, then of the projector (relies on multiple directory paths to camera calibration, ambient lighting of chessboard poses, gray-code projected stacks of chessboard poses)
First it does Zhang's method of calibration, then it uses projector calibration algorithm described in D. Moreno and G. Taubin, "Simple, Accurate, and Robust Projector-Camera Calibration," 2012 Second International Conference on 3D Imaging, Modeling, Processing, Visualization & Transmission, Zurich, Switzerland, 2012, pp. 464-471, doi: 10.1109/3DIMPVT.2012.77.
saves the stereo calibrated of camera intrinsic/distortion, projector intrinsic/distortion, R, T, E, F
# triangulate
uses the stereo calibrated. Decodes input image stack of a scene and triangulates with cv2
# util
miscellaneous, like decoding
# generatePatterns
create binary/gray/xor patterns to put on the projector. They are all .tiff files. 
