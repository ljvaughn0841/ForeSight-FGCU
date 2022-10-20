"""
Establishes a connection between the camera and the computer.

https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/

To run this file you must use the following:
 * pip install numpy
 * pip install opencv_python
"""

# Import Open CV
import cv2 as cv

# define a video capture object
vid = cv.VideoCapture(0)

# get a frame from the device
rep, frame = vid.read()

while True:

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    # cv.imshow('frame', frame)

    ######################
    # https://learnopencv.com/edge-detection-using-opencv/

    # Convert to graycsale
    img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv.GaussianBlur(img_gray, (3, 3), 0)

    # Canny Edge Detection
    edges = cv.Canny(image=img_blur, threshold1=100,
                      threshold2=200)  # Canny Edge Detection
    # Display Canny Edge Detection Image
    cv.imshow('Canny Edge Detection', edges)

    ##################

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()
