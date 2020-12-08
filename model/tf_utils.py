import os
import warnings

import GPUtil
import tensorflow as tf


class batch_norm:
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(
            x,
            decay=self.momentum,
            updates_collections=None,
            epsilon=self.epsilon,
            scale=True,
            is_training=train,
            scope=self.name,
        )


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def conv2d(
    input_,
    output_dim,
    k_h=5,
    k_w=5,
    d_h=2,
    d_w=2,
    stddev=0.02,
    padding="same",
    name="enc",
):
    with tf.variable_scope(name):
        return tf.layers.conv2d(
            input_, output_dim, [k_h, k_w], [d_h, d_w], padding=padding
        )


def deconv2d(
    input_,
    output_dim,
    k_h=5,
    k_w=5,
    d_h=2,
    d_w=2,
    stddev=0.02,
    padding="same",
    name="dec",
):
    with tf.variable_scope(name):
        return tf.layers.conv2d_transpose(
            input_, output_dim, [k_h, k_w], [d_h, d_w], padding=padding
        )


def set_gpu(device_id: str):
    # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)


def setup_gpu():
    # Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Get the first available GPU
    device_id = GPUtil.getAvailable(
        order="memory", limit=1, maxLoad=0.9, maxMemory=0.8, includeNan=False
    )[0]

    set_gpu(device_id)


def silence_deprication_warnings():
    try:
        import tensorflow.python.util.deprecation as deprecation

        deprecation._PRINT_DEPRECATION_WARNINGS = False
    except:
        pass
    try:
        from tensorflow.python.util import deprecation

        deprecation._PRINT_DEPRECATION_WARNINGS = False
    except:
        pass
    try:
        from tensorflow.python.util import module_wrapper as deprecation
    except ImportError:
        from tensorflow.python.util import deprecation_wrapper as deprecation
    deprecation._PER_MODULE_WARNING_LIMIT = 0
    warnings.filterwarnings("ignore", message=" The name tf")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message="\nThe TensorFlow contrib")
