"""
Establishes a connection between the camera and the computer.

https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/

To run this file you must use the following:
 * pip install numpy
 * pip install opencv_python
"""

# Import Open CV
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# define a video capture object
vid = cv.VideoCapture(0)

# Initializing Items for the Lucas-Kanade Optical Flow
# params for corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                           10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = vid.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None,
                            **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
# End of initializing items for optical flow


while True:

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    cv.imshow('input', frame)

    # Displaying the frame with canny edge detection in another window
    ######################
    # https://learnopencv.com/edge-detection-using-opencv/

    # Convert to grayscale
    img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv.GaussianBlur(img_gray, (3, 3), 0)

    # Canny Edge Detection
    edges = cv.Canny(image=img_blur, threshold1=100,
                     threshold2=200)

    # Display Canny Edge Detection Image
    cv.imshow('Canny Edge Detection', edges)

    # Lucas-Kanade Optical Flow Starts
    # https://www.geeksforgeeks.org/python-opencv-optical-flow-with-lucas-kanade-method/?ref=lbp
    frame_gray = cv.cvtColor(frame,
                             cv.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray,
                                          frame_gray,
                                          p0, None,
                                          **lk_params)

    # Select good points

    if p1 is None:
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray,
                                              frame_gray,
                                              p0, None, **lk_params)
        # clear the mask
        mask = np.zeros_like(old_frame)

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new,
                                       good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)),
                       color[i].tolist(), 2)

        frame = cv.circle(frame, (int(a), int(b)), 5,
                          color[i].tolist(), -1)

    img = cv.add(frame, mask)

    cv.imshow('Optical Flow', img)

    # Updating Previous frame and points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # END of Lucas Kankade Optical Flow

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()
