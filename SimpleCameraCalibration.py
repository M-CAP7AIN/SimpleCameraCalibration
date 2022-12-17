import numpy as np
import cv2 as cv
import glob
import json
import pickle

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# Matrix of Chessboard
rows = 6
cols = 7


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((rows*cols,3), np.float32)
objp[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


# Get List of Images from folder
print("Read Images")
images = glob.glob('./input_images/*.png')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (cols,rows), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (cols,rows), corners2, ret)
        cv.imshow('ChessBoard', img)
        
        key = cv.waitKey(20) & 0xFF
        if key == ord("q"):
            break
cv.destroyAllWindows()


# Get Calibration parameters
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


# Save Calibration parameters
print("Save the camera calibration results. (json)")
calib_result_pickle = {}
calib_result_pickle["mtx"] = mtx
calib_result_pickle["dist"] = dist

with open("camera_calib_pickle.p", 'wb') as f:
    pickle.dump(calib_result_pickle, f)

with open("camera_calib_pickle.json", 'w', encoding='utf-8') as f:
    json.dump(calib_result_pickle, fp=f, cls=NumpyEncoder)



# Read Camera Calibration parameters from file
# dist_pickle = pickle.load( open( "camera_calib_pickle.p", "rb" ) )
# mtx = dist_pickle["mtx"]
# dist = dist_pickle["dist"]



images = glob.glob('./input_images/*.png')
print("Read Images")
for fname in images:
    img = cv.imread(fname)
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imshow('Source', img)
    cv.imshow('Calibrate', dst)

    key = cv.waitKey(1000) & 0xFF
    if key == ord("q"):
        break

cv.destroyAllWindows()