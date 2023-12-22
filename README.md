# Image Upscaling with ESRGAN

This Python script uses the ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks) model from TensorFlow Hub to upscale images. The script resizes the images by a specified scale factor and then uses the ESRGAN model to enhance the quality of the resized images.

## How to Run

1. Install Python: Download and install the latest version of Python from the official website. Make sure to check the box that says "Add Python to PATH" during the installation process.

2. Install necessary libraries: Open Command Prompt and install the necessary libraries by running the following commands:
    ```shell
    pip install tensorflow
    pip install tensorflow-hub
    pip install pillow
    pip install opencv-python
    ```

3. Run the code: Save your code in a `.py` file and run it from the command line using the following command:
    ```shell
    python upscale.py
    ```

## What the Code Does

The script first resizes all the `.jpg` images in the current directory by a specified scale factor using OpenCV. It then uses the ESRGAN model to upscale the resized images, enhancing their quality. The upscaled images are saved with the prefix 'upscaled_'.

## Known Issues

The script throws an error when calling `output = model(image)`. The error message indicates that TensorFlow encountered an issue with the MKL (Math Kernel Library) when trying to execute a convolution operation. This could be due to a variety of reasons, such as incompatible versions of TensorFlow and MKL, or hardware constraints and i currently do not know how to get away with it.
