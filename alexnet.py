from skimage import io, transform
import numpy as np
import tensorflow as tf
# import cv2
from caffe_classes import class_names
import matplotlib.pyplot as plt

# data = np.load('./bvlc_alexnet.npy', encoding='bytes').item()
# print(data)
# print(class_names)


class AlexNet(object):
    def __init__(self, input_x, keep_prob, num_classes, skip_layer, weights_path='Default'):
        # Initialization the parameters
        self.keep_prob = keep_prob
        self.skip_layer = skip_layer
        if weights_path == 'Default':
            self.weights_path = './bvlc_alexnet.npy'
        else:
            self.weights_path = weights_path
        self.num_classes = num_classes
        # Create the AlexNet Network Define
        # self.create(input_x)

    def conv(self, x, ksize, depth_in, depth_out, stride,
             padding='SAME', name='conv', stddev=5e-2, const=0.0, groups=1):
        print('name is {} np.shape(input) {}'.format(name, np.shape(x)))

        convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride, stride, 1], padding=padding)

        with tf.variable_scope(name) as scope:
            # kernel = tf.get_variable('weights', shape=[ksize, ksize, depth_in/groups, depth_out],
            #                          initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
            # biases = tf.get_variable('biases', shape=[depth_out],
            #                          initializer=tf.constant_initializer(const))
            kernel = tf.get_variable('weights', shape=[ksize, ksize, depth_in/groups, depth_out])
            biases = tf.get_variable('biases', shape=[depth_out])

            if groups == 1:
                conv = convolve(x, kernel)
            else:
                input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
                weights_groups = tf.split(axis=3, num_or_size_splits=groups, value=kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, weights_groups)]

                conv = tf.concat(axis=3, values=output_groups)

            # conv = tf.nn.conv2d(x, kernel, [1, stride, stride, 1], padding=padding)
            # act = tf.nn.bias_add(conv, biases, name=scope.name)
            # bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
            bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
            relu = tf.nn.relu(bias, name=scope.name)
            return relu

    def max_pool(self, x, ksize, stride, padding, name):
        print('name is {} np.shape(input) {}'.format(name, np.shape(x)))
        with tf.variable_scope(name) as scope:
            pool = tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                                  padding=padding, name=scope.name)
            return pool

    def lrn(self, x, name):
        print('name is {} np.shape(input) {}'.format(name, np.shape(x)))
        with tf.variable_scope(name) as scope:
            lrn = tf.nn.local_response_normalization(
                x, depth_radius=2, bias=1, alpha=1e-05, beta=0.75, name=scope.name)
            return lrn

    def fc(self, x, depth_in, depth_out, name, relu=True, dropout=1.0, stddev=0.04, const=0.0):
        print('name is {} np.shape(input) {}'.format(name, np.shape(x)))
        with tf.variable_scope(name) as scope:
            # weight = tf.get_variable('weights', shape=[depth_in, depth_out],
            #                          initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
            # biases = tf.get_variable('biases', shape=[depth_out],
            #                          initializer=tf.constant_initializer(const))
            weight = tf.get_variable('weights', shape=[depth_in, depth_out], trainable=True)
            biases = tf.get_variable('biases', shape=[depth_out], trainable=True)
            act = tf.nn.xw_plus_b(x, weight, biases, name=scope.name)

            if relu is True:
                return tf.nn.relu(act)
            else:
                return act
            # return tf.nn.dropout(out, dropout)

    def inference(self, x):
        init_depth = int(x.shape[-1])
        # layer 1
        conv1 = self.conv(x, depth_in=init_depth, depth_out=96, ksize=11, stride=4, padding='VALID', name='conv1')
        norm1 = self.lrn(conv1, name='norm1')
        pool1 = self.max_pool(norm1, ksize=3, stride=2, name='pool1', padding='VALID')
        # layer 2
        conv2 = self.conv(pool1, depth_in=96, depth_out=256, ksize=5, stride=1, name='conv2', groups=2)
        norm2 = self.lrn(conv2, name='norm2')
        pool2 = self.max_pool(norm2, ksize=3, stride=2, name='pool2', padding='VALID')
        # layer 3
        conv3 = self.conv(pool2, depth_in=256, depth_out=384, ksize=3, stride=1, name='conv3')
        # layer 4
        conv4 = self.conv(conv3, depth_in=384, depth_out=384, ksize=3, stride=1, name='conv4', groups=2)
        # layer 5
        conv5 = self.conv(conv4, depth_in=384, depth_out=256, ksize=3, stride=1, name='conv5', groups=2)
        pool5 = self.max_pool(conv5, ksize=3, stride=2, name='pool5', padding='VALID')
        # layer 6
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = self.fc(flattened, depth_in=6*6*256, depth_out=4096, name='fc6')
        drop6 = tf.nn.dropout(fc6, self.keep_prob)
        # layer 7
        fc7 = self.fc(drop6, depth_in=4096, depth_out=4096, name='fc7')
        drop7 = tf.nn.dropout(fc7, self.keep_prob)
        # fc7 = self.fc(input=fc6, num_in=4096, num_out=4096, name='fc7', drop_ratio=1.0 - self.keep_prob, relu=True)
        # layer 8
        fc8 = self.fc(drop7, depth_in=4096, depth_out=self.num_classes, name='fc8', relu=False)
        return fc8

    # load pretrained weights
    def load_weights(self, session):
        weights_dict = np.load(self.weights_path, encoding='bytes').item()

        for op_name in weights_dict:
            if op_name not in self.skip_layer:
                with tf.variable_scope(op_name, reuse=True):
                    for data in weights_dict[op_name]:
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))


# x = tf.placeholder(tf.float32, [1, 227, 227, 3])

# imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)
imagenet_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)

image = io.imread('./images/forklift.jpg')
# convert to BGR mode as cv2 type
image = image[:, :, (2, 1, 0)]

image = transform.resize(image, (227, 227, 3), anti_aliasing=True, mode='constant')
image = image * 255

image = np.array(image) - imagenet_mean
# io.imshow(image[:, :, (2, 1, 0)] / 255.0)

image = image.astype(np.float32)
x = np.reshape(image, [1, 227, 227, 3])

# io.imshow(x)
# x = np.asarray([x], dtype=np.float32)

model = AlexNet(x, 1, 1000, skip_layer=[])

logits = model.inference(x)

softmax = tf.nn.softmax(logits)

label_id = tf.argmax(logits, 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model.load_weights(sess)
    prob = sess.run(softmax)

    # plt.figure()
    # plt.plot(prob[0].tolist(), label='test')
    # plt.show()

    probs = prob[0, np.argmax(prob)]
    class_name = class_names[np.argmax(prob)]

    print(class_name)

# parameter = np.load('./bvlc_alexnet.npy', encoding='bytes').item()
# print(parameter)

