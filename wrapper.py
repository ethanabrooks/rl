# import cv2
from functools import partial

import numpy as np
import scipy.misc
from PIL import Image


def resize(shape, scale,
           crop_top,
           crop_left,
           crop_right,
           crop_bottom):
    assert len(shape) >= 2
    shape = list(shape)
    shape[0] -= (crop_top + crop_bottom)
    shape[1] -= (crop_left + crop_right)
    for dim in range(2):
        shape[dim] *= scale
    return map(int, shape[:2])


def preprocess(scale,
               crop_top,
               crop_left,
               crop_right,
               crop_bottom,
               image):
    """
    :param image: array size [height, width, 3]
    :param scale: percent to shrink image in range (0., 1.]
    :return: resized grayscale image size [height*scale, width*scale, 1]
    """
    assert image.ndim == 3
    assert image.shape[2] == 3
    rgb_weighting = [0.299, 0.587, 0.114]

    crop_right = None if crop_right == 0 else -crop_right
    crop_bottom = None if crop_bottom == 0 else -crop_bottom

    cropped = image[crop_top:crop_bottom, crop_left:crop_right]

    gray = np.dot(cropped, rgb_weighting)
    return scipy.misc.imresize(gray, scale)
    # return cv2.resize(gray, (0, 0), fx=scale, fy=scale)


class SpaceWrapper:
    def __init__(self, shape):
        self.shape = shape


class EnvWrapper:
    def __init__(self, env,
                 scale=1.,
                 buffer_size=4,
                 crop_top=0,
                 crop_left=0,
                 crop_right=0,
                 crop_bottom=0):
        self.__dict__.update(env.__dict__)
        buffer_size = int(buffer_size)
        self._env = env
        self._preprocess = partial(preprocess,
                                   scale, crop_top, crop_left, crop_right, crop_bottom)
        shape = resize(self.observation_space.shape, scale,
                       crop_top, crop_left, crop_right, crop_bottom)
        self._image_buffer = [np.zeros(shape) for _ in range(buffer_size)]
        shape[0] *= buffer_size
        self.observation_space = SpaceWrapper(shape)

    def _updated_concat_images(self, new_image):
        self._image_buffer.append(self._preprocess(new_image))
        self._image_buffer.pop(0)
        concat_images = np.vstack(self._image_buffer)
        for dim1, dim2 in zip(concat_images.shape, self.observation_space.shape):
            assert dim1 == dim2, '{} vs. {}'.format(concat_images.shape, self.observation_space.shape)
        return concat_images

    def reset(self):
        return self._updated_concat_images(self._env.reset())

    def step(self, action):
        step_result = list(self._env.step(action))
        step_result[0] = self._updated_concat_images(step_result[0])
        # if step_result[2]:
        #     fromarray = Image.fromarray(step_result[0])
        #     fromarray.show()
        #     print('waiting')
        #     try:
        #         while True:
        #             pass
        #     except KeyboardInterrupt:
        #         print('resuming')
        return step_result
