#!/usr/bin/env python3

from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage import rotate, shift
import tensorflow as tf
import numpy as np
import pathlib
import random


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def load_data(dat_dir, dim, mapping, normalize=True, dim_new=None, binary=False):
    files = list(dat_dir)
    samples_per_cat = 10000
    image_count = len(files) * samples_per_cat
    if dim_new is None:
        images = np.empty((image_count, dim, dim))
    else:
        images = np.empty((image_count, dim_new, dim_new))
    labels = np.empty(image_count, dtype=np.int32)
    
    for i, f in enumerate(files):
        tmp = np.load(f)
        tmp = tmp.f.arr_0
        for j, img in enumerate(tmp):
            labels[i * samples_per_cat + j] = int(mapping[f.name.split(".")[0]])
            if dim_new is not None:
                assert dim_new > dim
                images[i * samples_per_cat + j] = upsample(dim_new, img.reshape(dim, dim))
            else:
                images[i * samples_per_cat + j] = img.reshape(dim, dim)

    if normalize:
        images = (images / 127.5) - 1

    if binary:
        images[images >= .5] = 1.
        images[images < .5] = 0.

    return images, image_count, samples_per_cat, labels


def upsample(dim_new, arr):
    dim_old = np.shape(arr)[0]
    assert np.shape(arr)[0] == np.shape(arr)[1]
    x = np.arange(dim_old)
    f = interpolate.interp2d(x, x, arr, kind='linear')
    xnew = np.linspace(0, dim_old, dim_new)
    ynew = np.linspace(0, dim_old, dim_new)
    return f(xnew, ynew)


def load_for_cnn(dat_path, dim, d_labels, split, channel_axes=1, dim_new=None, test=False, binary=False):
    """
    Inputs:
        dat_path: pathlib path to train/test directory
                  e.g. pathlib.Path('images/').glob('train/*.npz')
        dim: dimension of 2D image files
             e.g. 28 for (28, 28) image file
        d_labels: dict mapping from string file labels to integers
                  e.g. d = {"lollipop":0, "canoe":1, "penguin":2, "eyeglasses":3, "apple":4}
                  TODO--this shouldn't need to be an argument! should generate internally
                        then return the mapping for future use
        split: tuple specifying train and validation fraction
        channel_axes: number of channel axes (1 for greyscale and 3 for rgb)
        dim_new: for use in transfer learning examples that require a particular input size
                 automatically upsamples to dim_new using linear interpolation
                 NB: only upsamples, downsampling is not supported
        test: display subset of images with labels
    """
    images, image_count, samples_per_cat, labels = load_data(dat_path, dim, d_labels, dim_new=dim_new, binary=binary)
    assert channel_axes == 0 or channel_axes == 1 or channel_axes == 3
    if dim_new is None:
        dim_new = dim
    if channel_axes == 3:
        images = np.stack((images, images, images), axis=3)
    elif channel_axes == 1:
        images = images.reshape(image_count, dim_new, dim_new, 1)

    if test:
        id_labels = {0:"lollipop", 1:"canoe", 2:"penguin", 3:"eyeglasses", 4:"apple", 5:"moon", 6:"cup"}
        for i in range(0, image_count, int(image_count / 20)):
            plt.imshow(images[i].reshape(dim_new, dim_new))
            plt.title(id_labels[labels[i]])
            plt.show()

    train, test, train_labels, test_labels = train_test_split(images, labels, train_size=split[0])
    valid, test, valid_labels, test_labels = train_test_split(test, test_labels, train_size=split[1])

    return train, tf.one_hot(train_labels, depth=7), valid, tf.one_hot(valid_labels, depth=7), test, tf.one_hot(test_labels, depth=7)


def shuffle(dataset, batch_size=None, tensor=False, buffer_size=1000):
    """
    Inputs:
        dataset: output of load_for_cnn
        batch_size: relevant only if tensor is True
        tensor: specifies datatype (for false, input is numpy)
        buffer_size: for shuffling data--should be equal to the total number of samples for
                     consistency with numpy implementation
    Outputs:
       shuffled dataset
    """
    if tensor:
        return dataset.shuffle(buffer_size).batch(batch_size)
    return unison_shuffled_copies(dataset[0], dataset[1])

    
if __name__ == "__main__":
    d = {"lollipop":0, "canoe":1, "penguin":2, "eyeglasses":3, "apple":4, "moon":5, "cup":6}
    d_test = pathlib.Path('object_files/').glob('*.npz')
    load_for_cnn(d_test, 28, d, (0.7, 0.25), channel_axes=1, dim_new=None, test=False)


class ImageDataGenerator(object):
    def __init__(self, data, labels, batch_size, channel_axes):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.dim = len(self.data[0])
        self.channel_axes = channel_axes
        assert(len(np.shape(self.data)) == 3)  # make sure we haven't reshaped yet

    def flow(self):
        return BaseIterator(self.data, self.labels, self, self.batch_size, self.channel_axes)
    
    def random_transform(self, x):
        # There are only a few transformations that make sense here--rotations of <45
        # degrees in either direction and horizontal flips, and shifts by at most 10%
        # of the image width
        theta = random.uniform(-5, 5)
        x = rotate(x, theta, reshape=False, mode='constant', cval=-1.0)  # the default is white anyway
        if random.random() > 0.5:
            x = np.fliplr(x)
        shift_x = random.randint(-self.dim // 20, self.dim // 20)
        shift_y = random.randint(-self.dim // 20, self.dim // 20)
        x = shift(x, (shift_x, shift_y), mode='constant', cval=-1.0)

        # Finally, reshape to the correct number of channel axes
        if self.channel_axes == 1:
            return np.reshape(x, (len(x), len(x), 1))
        elif self.channel_axes == 3:
            return np.stack((x, x, x), axis=3)
        return x


class Iterator(object):
    def __init__(self, batch_size, data_size):
        self.batch_size = batch_size
        self.data_size = data_size
        self.remaining_indices = np.arange(data_size)
        np.random.shuffle(self.remaining_indices)
        self.index_generator = self._flow_index()

    def _flow_index(self):
        while True:
            if len(self.remaining_indices) == 0:
                self.reset()
            current_batch_size = min(self.batch_size, len(self.remaining_indices))
            indices = np.random.choice(self.remaining_indices, current_batch_size, replace=False)
            self.remaining_indices = np.setxor1d(indices, self.remaining_indices)
            yield(indices, current_batch_size)

    def reset(self):
        self.remaining_indices = np.random.shuffle(np.arange(self.data_size))

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    
class BaseIterator(Iterator):
    def __init__(self, data, label, image_data_generator, batch_size, channel_axes):
        self.image_data_generator = image_data_generator
        self.data = data
        self.label = np.array(label)
        self.batch_size = batch_size
        self.channel_axes = channel_axes
        self.dim = len(self.data[0])
        data_size = len(data)
        self.shape = (self.batch_size, self.dim, self.dim)
        super(BaseIterator, self).__init__(batch_size, data_size)

    def next(self):
        index_array, current_batch_size = next(self.index_generator)
        self.shape = (current_batch_size, self.dim, self.dim)

        batch_x = self.data[index_array]
        batch_y = self.label[index_array]
        new_batch_x = np.empty((current_batch_size, self.dim, self.dim, self.channel_axes))

        for i in range(len(batch_x)):
            new_batch_x[i] = self.image_data_generator.random_transform(batch_x[i]) 

        print(np.shape(new_batch_x))
        return new_batch_x, batch_y
