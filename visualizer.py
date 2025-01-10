import cv2
import numpy as np

class Visualizer():

    def __init__(self, title):
        self.title = title
        self.recorder = None

    def stream(self, frame):
        self.frame = frame

    def show(self):

        self.frame = cv2.resize(self.frame, (640, 520))

        cv2.imshow(self.title, self.frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.recorder.release()
            return 1
        
        return 0
    
    def depict_vectorfield(self, field, range_field = (0, 1, 0, 1)):

        size_field = field.shape

        scale = self.frame.shape[0] / size_field[0]

        d1 = int(range_field[0] * size_field[0])
        d2 = int(range_field[1] * size_field[0])
        d3 = int(range_field[2] * size_field[1])
        d4 = int(range_field[3] * size_field[1])

        for i in np.arange(d1, d2, 5):
            for j in np.arange(d3, d4, 5):

                # vector size
                a0 = (field[i, j, 0]) * 3
                a1 = (field[i, j, 1]) * 3

                # vector position
                x = int(j * scale)
                y = int(i * scale)

                cv2.arrowedLine(self.frame, (x ,y), (x + int(a0), y + int(a1)), (50,50,250), 1)

        return self.frame
    
    def depict_hue(self, motion_vectors):
        frame_hue = self.flow_to_hue_image(motion_vectors)

        if frame_hue is not None:
            [h, w] = self.frame.shape[:2]
            frame_hue = cv2.resize(frame_hue, (w, h))
            self.frame = cv2.vconcat([self.frame, frame_hue])
    
    @staticmethod
    def flow_to_hue_image(flow):
        # Create an image where the motion vector angle is
        # mapped to a color hue, and intensity is proportional
        # to vector's magnitude

        magnitude, angle = cv2.cartToPolar(flow[:,:,0], flow[:,:,1], angleInDegrees=True)
    
        clip = 5.0
        cv2.threshold(magnitude, clip, clip, cv2.THRESH_TRUNC, magnitude)
    
        # build the hsv image
        hsv = np.ndarray([flow.shape[0], flow.shape[1], 3], np.float32)
        hsv[:,:,0] = angle
        hsv[:,:,1] = np.ones((angle.shape[0], angle.shape[1]), np.float32)
        hsv[:,:,2] = magnitude / clip
    
        # Convert HSV to BGR8
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return np.uint8(bgr*255)

    def record(self, path):
        if self.recorder == None:
            ### Recorder ###
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi files
            fps = 30.0  # Frames per second

            frame_shape = np.shape(self.frame)
            frame_width = frame_shape[1]
            frame_height = frame_shape[0]

            # Create VideoWriter object to save the video
            self.recorder = cv2.VideoWriter(path + '.avi', fourcc, fps, (frame_width, frame_height))
        
        self.recorder.write(self.frame)