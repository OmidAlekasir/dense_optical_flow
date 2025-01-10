import vpi
import numpy as np

class DenseOpticalFlow():

    def __init__(self, gridSize, pyramid_level):

        self.f0 = None

        self.gridSize = gridSize
        self.pyramid_level = pyramid_level

        self.quality = vpi.OptFlowQuality.LOW

    def get_flow(self, frame):

        self.frame = frame
        self.__calculate_flow()

        return self.flow
    
    def __calculate_flow(self):
        f1 = self.__convert_to_vpiframe()
        
        if self.f0 == None:
            self.f0 = f1
        
        motion_vector = self.__calculate_motionvector(self.f0 ,f1)
        self.f0 = f1

        self.flow = self.__convert_flow(motion_vector)

    def __convert_to_vpiframe(self):

        vpiframe = vpi.asimage(self.frame, vpi.Format.BGR8) \
                    .convert(vpi.Format.Y8_ER, backend=vpi.Backend.CUDA) \
                    .gaussian_pyramid(self.pyramid_level, backend=vpi.Backend.CUDA) \
                    .convert(vpi.Format.Y8_ER_BL, backend=vpi.Backend.VIC)
        
        return vpiframe
    
    def __calculate_motionvector(self, f0 ,f1):
        backend = vpi.Backend.OFA
        with backend:
            motion_vectors = vpi.optflow_dense(f0, f1, quality = self.quality, gridsize = self.gridSize)

        return motion_vectors
    
    def __convert_flow(self, mv):
        with mv.rlock_cpu() as data:
            flow = np.float32(data)/(1<<5)

        return flow