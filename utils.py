"""
StarGAN v2 TensorFlow Implementation
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import numpy as np
import os
import cv2

import tensorflow as tf
import random
from glob import glob

class Image_data:

    def __init__(self, img_size, channels, dataset_path, augment_flag, batch_size, latent_dim):
        self.img_height = img_size
        self.img_width = img_size
        self.channels = channels
        self.augment_flag = augment_flag

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.latent_dim = latent_dim


    def parse_function(self, example):

        feature_description = {
            'img': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'domain': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        }
        example_dict = tf.io.parse_single_example(example, feature_description)
        return tf.io.parse_tensor(example_dict['img'], out_type=tf.float32), example_dict['domain']

    def read_function(self, filename, domain):

        x = tf.io.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels, dct_method='INTEGER_ACCURATE')
        img = tf.image.resize(x_decode, [self.img_height, self.img_width])
        img = preprocess_fit_train_image(img)

        return img, domain

    def image_processing(self, img, domain):

        seed = random.randint(0, 2 ** 31 - 1)
        condition = tf.greater_equal(tf.random.uniform(shape=[], minval=0.0, maxval=1.0), 0.5)

        augment_height_size = self.img_height + (30 if self.img_height == 256 else int(self.img_height * 0.1))
        augment_width_size = self.img_width + (30 if self.img_width == 256 else int(self.img_width * 0.1))

        img = tf.cond(pred=condition,
                      true_fn=lambda : augmentation(img, augment_height_size, augment_width_size, seed),
                      false_fn=lambda : img)

        return img, domain

    def inject_z(self, img_batch, domain_batch):

        z_vals = tf.random.normal([self.batch_size, self.latent_dim])
        return img_batch, domain_batch, z_vals

def build_filename_list(dataset_path, domain_list):
    # domain_list = ['tiger', 'cat', 'dog', 'lion']

    images, domains = [], []
    for idx, domain in enumerate(domain_list):
        image_list = glob(os.path.join(dataset_path, domain) + '/*.png') + glob(os.path.join(dataset_path, domain) + '/*.jpg')
        domain_list = [[idx]] * len(image_list)  # [ [0], [0], ... , [0] ]

        images.extend(image_list)
        domains.extend(domain_list)

    return images, domains

def adjust_dynamic_range(images, range_in, range_out, out_dtype):
    scale = (range_out[1] - range_out[0]) / (range_in[1] - range_in[0])
    bias = range_out[0] - range_in[0] * scale
    images = images * scale + bias
    images = tf.clip_by_value(images, range_out[0], range_out[1])
    images = tf.cast(images, dtype=out_dtype)
    return images

def preprocess_fit_train_image(images):
    images = adjust_dynamic_range(images, range_in=(0.0, 255.0), range_out=(-1.0, 1.0), out_dtype=tf.dtypes.float32)
    return images

def postprocess_images(images):
    images = adjust_dynamic_range(images, range_in=(-1.0, 1.0), range_out=(0.0, 255.0), out_dtype=tf.dtypes.float32)
    images = tf.cast(images, dtype=tf.dtypes.uint8)
    return images

def load_images(image_path, img_size, img_channel):
    x = tf.io.read_file(image_path)
    x_decode = tf.image.decode_jpeg(x, channels=img_channel, dct_method='INTEGER_ACCURATE')
    img = tf.image.resize(x_decode, [img_size, img_size])
    img = preprocess_fit_train_image(img)

    return img

def augmentation(image, augment_height, augment_width, seed):
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize(image, [augment_height, augment_width])
    image = tf.image.random_crop(image, ori_image_shape, seed=seed)
    return image

def load_test_image(image_path, img_width, img_height, img_channel):

    if img_channel == 1 :
        img = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
    else :
        img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, dsize=(img_width, img_height))

    if img_channel == 1 :
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
    else :
        img = np.expand_dims(img, axis=0)

    img = img/127.5 - 1

    return img

def save_images(images, size, image_path):
    # size = [height, width]
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return ((images+1.) / 2) * 255.0


def imsave(images, size, path):
    images = merge(images, size)
    images = cv2.cvtColor(images.astype('uint8'), cv2.COLOR_RGB2BGR)

    return cv2.imwrite(path, images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def return_images(images, size) :
    x = merge(images, size)

    return x

def check_folder(path_name):
    if not tf.io.gfile.exists(path_name):
        tf.io.gfile.makedirs(path_name)
    return path_name


def str2bool(x):
    return x.lower() in ('true')

def pytorch_xavier_weight_factor(gain=0.02, uniform=False) :

    factor = gain * gain
    mode = 'fan_avg'

    return factor, mode, uniform

def pytorch_kaiming_weight_factor(a=0.0, activation_function='relu') :

    if activation_function == 'relu' :
        gain = np.sqrt(2.0)
    elif activation_function == 'leaky_relu' :
        gain = np.sqrt(2.0 / (1 + a ** 2))
    elif activation_function =='tanh' :
        gain = 5.0 / 3
    else :
        gain = 1.0

    factor = gain * gain
    mode = 'fan_in'

    return factor, mode

def automatic_gpu_usage() :
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def multiple_gpu_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Create 2 virtual GPUs with 1GB memory each
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096),
                 tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)