import cv2
from dense_optical_flow import DenseOpticalFlow
from visualizer import Visualizer

cap = cv2.VideoCapture('vid.mp4')
dof = DenseOpticalFlow(gridSize=4, pyramid_level=3)
vis = Visualizer('VPI dense optical flow')

if __name__ == '__main__':

    # Start capturing video frames
    while True:
        # Read a frame
        ret, frame = cap.read()
        
        # If frame is read correctly, ret will be True
        if not ret:
            print("Error: Could not read frame.")
            break

        # calculate flow
        flow = dof.get_flow(frame)
        
        # Display the resulting frame
        vis.stream(frame)
        vis.depict_vectorfield(flow)
        vis.depict_hue(flow)
        vis.record('result')
        if vis.show():
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()