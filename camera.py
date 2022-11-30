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
import torch
from matplotlib import pyplot as plt


def edge_detection(frame):
    """
    Uses Canny Edge Detection to produce a greyscale image of the
    frame with white as an edge and black as no edge.

    https://learnopencv.com/edge-detection-using-opencv/
    """
    # Convert to grayscale
    img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv.GaussianBlur(img_gray, (3, 3), 0)

    # Canny Edge Detection
    # Thresholds can be tweaked in order to increase accuracy.
    edges = cv.Canny(image=img_blur, threshold1=100,
                     threshold2=200)

    # Displaying the frame with canny edge detection in another window
    cv.imshow('Canny Edge Detection', edges)


class OpticalFlow:

    def __init__(self, vid):
        # Initializing Items for the Lucas-Kanade Optical Flow
        # params for corner detection
        self.feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)

        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(
                         cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                         10, 0.03))

        # Create some random colors
        self.color = np.random.randint(0, 255, (100, 3))

        # Take first frame and find corners in it
        self.ret, self.old_frame = vid.read()
        self.old_gray = cv.cvtColor(self.old_frame, cv.COLOR_BGR2GRAY)
        self.p0 = cv.goodFeaturesToTrack(self.old_gray, mask=None,
                                    **self.feature_params)

        # Create a mask image for drawing purposes
        self.mask = np.zeros_like(self.old_frame)

    def lk_optical_flow(self, frame):
        """
        Processing for Lucas Kankade Optical Flow
        """
        # Lucas-Kanade Optical Flow Starts
        # https://www.geeksforgeeks.org/python-opencv-optical-flow-with-lucas-kanade-method/?ref=lbp
        frame_gray = cv.cvtColor(frame,
                                 cv.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(self.old_gray,
                                              frame_gray,
                                              self.p0, None,
                                              **self.lk_params)

        # Select good points

        if p1 is None:
            self.p0 = cv.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
            p1, st, err = cv.calcOpticalFlowPyrLK(self.old_gray,
                                                  frame_gray,
                                                  self.p0, None, **self.lk_params)
            # clear the mask
            self.mask = np.zeros_like(self.old_frame)

        good_new = p1[st == 1]
        good_old = self.p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new,
                                           good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            self.mask = cv.line(self.mask, (int(a), int(b)), (int(c), int(d)),
                           self.color[i].tolist(), 2)

            frame = cv.circle(frame, (int(a), int(b)), 5,
                              self.color[i].tolist(), -1)

        img = cv.add(frame, self.mask)

        cv.imshow('Optical Flow', img)

        # Updating Previous frame and points
        self.old_gray = frame_gray.copy()
        self.p0 = good_new.reshape(-1, 1, 2)


class Midas_Depth:
    """
    Depth estimation using MiDaS.

CODE REFERENCES:
    MiDaS:
        @ARTICLE {Ranftl2022,
        author  = "Ren\'{e} Ranftl and Katrin Lasinger and David Hafner
                    and Konrad Schindler and Vladlen Koltun",
        title   = "Towards Robust Monocular Depth Estimation: Mixing Datasets
                    for Zero-Shot Cross-Dataset Transfer",
        journal = "IEEE Transactions on Pattern Analysis and
                    Machine Intelligence",
        year    = "2022",
        volume  = "44",
        number  = "3"
        }

    Tutorial Video:
        Nicolai Nielsen - Computer Vision & AI
        https://youtu.be/jid-53uPQr0
    """
    def __init__(self,):
        # Loads a model for depth estimation
        # MiDaS v2.1 - Small (the lowest accuracy, the highest inference speed)
        self.model_type = "MiDaS_small"

        # Loads the model from the github.
        # https://github.com/isl-org/MiDaS
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)

        self.midas.eval()

        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        self.transform = self.midas_transforms.small_transform

    def depth_estimation(self, img):

        # Transforms the images from BGR format to RGB
        # BGR is cv2's default but MiDaS uses RGB
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Apply input transforms
        input_batch = self.transform(img).to(self.device)

        # Prediction and resize to original resolution
        # torch.no_grad() disables gradient calculation
        with torch.no_grad():
            # We pass it our input image
            prediction = self.midas(input_batch)

            # Resizing to Original Resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # takes the prediction puts it on the cpu in a numpy array format
        # for post-processing
        depth_map = prediction.cpu().numpy()
        # Uses minmax normalization on the image
        # ( sets the lowest values to zero and highest to 1 )
        depth_map = cv.normalize(depth_map, None, 0, 1,
                                  norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)

        # converting image back into BGR for open cv
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        # converting values from '0 to 1' to '0 to 255' (RGB color range)
        depth_map = (depth_map * 255).astype(np.uint8)

        # applies a color map to the image for better visualization
        # depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)

        # displays the img
        cv.imshow('Image', img)
        # displays the depth map
        cv.imshow('Depth Map', depth_map)

        edge_map = cv.cvtColor(depth_map, cv.COLOR_RGB2BGR)

        edge_detection(edge_map)


def main():
    # define a video capture object
    vid = cv.VideoCapture(0)

    lk = OpticalFlow(vid)

    md = Midas_Depth()

    while True:

        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        # Display the resulting frame
        #cv.imshow('input', frame)

        #edge_detection(frame)

        md.depth_estimation(frame)

        #lk.lk_optical_flow(frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
