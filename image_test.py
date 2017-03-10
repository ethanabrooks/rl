import numpy as np
import time

from model_zoo import conv_layer
import tensorflow as tf
from wrapper import resize, preprocess

height = 100
width = 100
buffer_size = 4
scale = .3

images = [np.random.normal(size=(height, width, 3)) for _ in range(buffer_size)]

with tf.Session() as sess:
    sess.run(tf.constant(0))

print('Method 1: use scipy')
start = time.time()
concat_images = np.vstack([preprocess(image, scale) for image in images])
image_ph = tf.placeholder(tf.float32, concat_images.shape)
inputs = tf.expand_dims(tf.expand_dims(image_ph, 0), 3)
output = conv_layer(0, inputs, 16, 4, scope='1')
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    sess.run(output, {image_ph: concat_images})
print(time.time() - start)


print('Method 2: use tf in one session')
start = time.time()
images_ph = []
preprocessed = []
for image in images:
    placeholder = tf.placeholder(tf.float32, image.shape)
    images_ph.append(placeholder)
    grayscale = tf.image.rgb_to_grayscale(placeholder)
    new_size = list(image.shape[:2])
    for dim in range(2):
        new_size[dim] *= scale
    resized = tf.image.resize_images(grayscale, tf.to_int32(tf.constant(new_size)))
    preprocessed.append(resized)
concat = tf.concat(preprocessed, 0)
inputs = tf.expand_dims(concat, 0)
output = conv_layer(0, inputs, 16, 4, scope='2')
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    sess.run(output, dict(zip(images_ph, images)))
print(time.time() - start)

# print('Method 3: use tf in two session')
# start = time.time()
# images_ph = []
# preprocessed = []
# for image in images:
#     placeholder = tf.placeholder(tf.float32, image.shape)
#     images_ph.append(placeholder)
#     grayscale = tf.image.rgb_to_grayscale(placeholder)
#     new_size = image.shape[:2]
#     for dim in range(32):
#         new_size[dim] *= scale
#     resized = tf.image.resize_images(grayscale, new_size)
#     preprocessed.append(resized)
#
# with tf.Session() as sess:
#     concat = sess.run(concat(preprocessed, 0), dict(*zip(images_ph, images)))
#     concat_ph = tf.placeholder(tf.float32, concat.shape)
#     output = conv_layer(0, concat, 16, 4, scope='3')
#     tf.global_variables_initializer().run()
#     sess.run(output, {concat_ph: concat})
# print(time.time() - start)


