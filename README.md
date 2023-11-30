# Deep Canny Edge Detector implemented in Pytorch

HUGE respect to Kornia and the team for their briliant implemetation of Canny algorithm and many other computer vision/ image processing algorithms.

## Features:
- Light-weight: Inspired by Known Operator Learning, this network replicate well-known Canny Edge Detection algorithm with the ability to learn custom kernel to adapt to specific cases. It has very few trainable parameters, which some can be merged with the standard non-trainable parameters for faster inference time. 
- Fast inference time: For converting to TFLite format, with the input size of (256,256) and number of hidden channels of 8, this can execute in ~67ms (tested on Core i7-10700K) compared to other DNN.
- Mobile conversion available: With the light-weight ability, it can be converted to any format that support inference or even training on edge devices. All unsupported operators are replaced with equivalent or a good approximator.

## Results:

## References:
[1] Wittmann, J., & Herl, G. (2023). Canny-Net: Known Operator Learning for Edge Detection. 12th Conference on Industrial Computed Tomography (iCT) 2023, 27 February - 2 March 2023 in Fürth, Germany. e-Journal of Nondestructive Testing Vol. 28(3). https://doi.org/10.58286/27751 

[2] https://discuss.pytorch.org/t/trying-to-train-parameters-of-the-canny-edge-detection-algorithm/154517 

[3] https://kornia.readthedocs.io/en/latest/_modules/kornia/filters/canny.html#canny
