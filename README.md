# dense_optical_flow
This repository provides a simple implementation and usage guide for NVIDIA's dense optical flow algorithm. Dense optical flow is a technique used in computer vision to compute the motion of pixels or small regions between two consecutive frames in a video sequence.

## Getting Started
1. Ensure you have an NVIDIA GPU installed and the CUDA toolkit is properly configured on your system.
2. Clone this repository to your local machine using the following command:
```bash
git clone https://github.com/OmidAlekasir/dense_optical_flow.git
```
3. Install necessary dependencies such as OpenCV and CUDA libraries. You can use pip for Python dependencies:
4. Navigate to the cloned repository and run the example script. This will demonstrate how to compute dense optical flow between two frames:
```bash
cd dense_optical_flow
python main.py
```
## Usage
* Input: The script expects a file named `vid.mp4`.
* Output: The output will be a visualization of the optical flow of the input video. Then the visualization will be saved as an `avi` file.

## NVIDIA Dense Optical Flow
This repository utilizes NVIDIA's hardware acceleration for efficient computation of dense optical flow. The algorithm is optimized for performance on NVIDIA GPUs, making it suitable for real-time video processing applications.
The tests have been executed on Jetson, demonstrating its capability for embedded systems. For more information, please refer to [this web page](https://docs.nvidia.com/vpi/algo_optflow_dense.html).