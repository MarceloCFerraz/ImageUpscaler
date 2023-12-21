import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image # Pillow
import cv2 # opencv-python
import os

# Not able to run this code with my laptop (check error below)
# 2023-12-21 13:49:45.315092: W tensorflow/core/framework/op_kernel.cc:1839] OP_REQUIRES failed at mkl_conv_grad_input_ops.cc:548 : ABORTED: Operation received an exception:Status: 1, message: could not create a primitive, in file tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc:545
# Traceback (most recent call last):
#   File "X:\[...]\upscale.py", line 76, in <module>
#     output = model(image)
#              ^^^^^^^^^^^^
#   File "X:\[...]\venv\Lib\site-packages\tensorflow\python\saved_model\load.py", line 816, in _call_attribute
#     return instance.__call__(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "X:\[...]\venv\Lib\site-packages\tensorflow\python\util\traceback_utils.py", line 153, in error_handler
#     raise e.with_traceback(filtered_tb) from None
#   File "X:\[...]\venv\Lib\site-packages\tensorflow\python\eager\execute.py", line 53, in quick_execute
#     tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# tensorflow.python.framework.errors_impl.AbortedError: Graph execution error:

# Detected at node StatefulPartitionedCall/StatefulPartitionedCall/rrdb_net/StatefulPartitionedCall/conv2d_transpose_1/StatefulPartitionedCall/conv2d_transpose defined at (most recent call last):
# <stack traces unavailable>
# Operation received an exception:Status: 1, message: could not create a primitive, in file tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc:545
#          [[{{node StatefulPartitionedCall/StatefulPartitionedCall/rrdb_net/StatefulPartitionedCall/conv2d_transpose_1/StatefulPartitionedCall/conv2d_transpose}}]] [Op:__inference_restored_function_body_23295]

# Load the ESRGAN models
model = hub.load('https://tfhub.dev/captain-pool/esrgan-tf2/1')

def preprocess_image(image_path):
    """Loads image from path and preprocesses to make it model ready"""
    if not os.path.isfile(image_path):
        print(f"File {image_path} not found.")
        return None
    hr_image = tf.image.decode_image(tf.io.read_file(image_path))
    # If PNG, remove the alpha channel. The model only supports images with 3 color channels.
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[...,:-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)

def save_image(image, filename):
    """
    Saves unscaled Tensor Images.
    Args:
    image: 3D image tensor. [height, width, channels]
    filename: Name of the file to save to.
    """
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save("%s.jpg" % filename)
    print("Saved as %s.jpg" % filename)

def resize_image(filename, scale_percent):
    # Load img
    img = cv2.imread(filename)

    # Calculate the new image dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize the image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)

    # Save the upscaled image
    cv2.imwrite("resized_" + filename, resized)

# Directory containing the images
directory = os.getcwd()

# Scale factor
scale_percent = 150 # percent of original size, can be changed to your desired value

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".jpg") and (not "resized_" in filename or not "upscaled_" in filename):
        print(f"Resizing {filename}")

        # Resize the image
        resize_image(filename, scale_percent)
        filename = "resized_" + filename

        print(f"Upscaling {filename}")

        # Preprocess the image
        image = preprocess_image(filename)
        if image is None:
            continue

        # Run the model
        output = model(image)

        # Save the result as upscaled_{filename}.jpg
        save_image(output[0, ...], f'{filename.replace("resized_", "upscaled_").replace(".jpg", "")}')
