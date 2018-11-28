'''
    For tensorflow dimensional testing
    reference for check dimension :  https://hackernoon.com/understanding-yolo-f5a74bbc7967
'''
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def conv2d(x, W, strides):
    return tf.nn.conv2d(x, W, strides=[1, strides[0], strides[1], 1], padding='SAME')

def max_pool(x, kernel_size=[2,2], strides=[2,2]):
    return tf.nn.max_pool(x, ksize=[1, kernel_size[0], kernel_size[1], 1],
                                strides=[1, strides[0], strides[1], 1], padding='SAME')

def conv_layer(input, shape, strides=[1,1]):
    W = weight_variable(shape)
    b = weight_variable([shape[3]])
    return tf.nn.leaky_relu(tf.add(conv2d(input, W, strides=strides), b))

def fc_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = weight_variable([size])
    return tf.add(tf.matmul(input, W), b)

size_width = 448
size_height = 448

feed_x = tf.truncated_normal(shape=[2, size_width, size_height, 3], stddev=0.1)

### start important part
x = tf.placeholder(tf.float32, shape=[2, 448, 448, 3], name='input_x')

conv1 = conv_layer(x, shape=[7, 7, 3, 64], strides=[2, 2])	# shape = [size, size, old_channel, new_channel]
max_pool1 = max_pool(conv1, kernel_size=[2, 2], strides=[2, 2])

conv2 = conv_layer(max_pool1, shape=[3, 3, 64, 192], strides=[1, 1])
max_pool2 = max_pool(conv2, kernel_size=[2, 2], strides=[2, 2])

conv3 = conv_layer(max_pool2, shape=[1, 1, 192, 128], strides=[1, 1])
conv4 = conv_layer(conv3, shape=[3, 3, 128, 256], strides=[1, 1])
conv5 = conv_layer(conv4, shape=[1, 1, 256, 256], strides=[1, 1])
conv6 = conv_layer(conv5, shape=[1, 1, 256, 512], strides=[1, 1])

max_pool3 = max_pool(conv6, kernel_size=[2, 2], strides=[2, 2])

conv7 = conv_layer(max_pool3, shape=[1, 1, 512, 256], strides=[1, 1])
conv8 = conv_layer(conv7, shape=[3, 3, 256, 512], strides=[1, 1])
conv9 = conv_layer(conv8, shape=[1, 1, 512, 256], strides=[1, 1])
conv10 = conv_layer(conv9, shape=[3, 3, 256, 512], strides=[1, 1])
conv11 = conv_layer(conv10, shape=[1, 1, 512, 256], strides=[1, 1])
conv12 = conv_layer(conv11, shape=[3, 3, 256, 512], strides=[1, 1])
conv13 = conv_layer(conv12, shape=[1, 1, 512, 256], strides=[1, 1])
conv14 = conv_layer(conv13, shape=[3, 3, 256, 512], strides=[1, 1])
conv15 = conv_layer(conv14, shape=[1, 1, 512, 512], strides=[1, 1])
conv16 = conv_layer(conv15, shape=[3, 3, 512, 1024], strides=[1, 1])

max_pool4 = max_pool(conv16, kernel_size=[2, 2], strides=[2, 2])

conv17 = conv_layer(max_pool4, shape=[1, 1, 1024, 512], strides=[1, 1])
conv18 = conv_layer(conv17, shape=[3, 3, 512, 1024], strides=[1, 1])
conv19 = conv_layer(conv18, shape=[1, 1, 1024, 512], strides=[1, 1])
conv20 = conv_layer(conv19, shape=[3, 3, 512, 1024], strides=[1, 1])
conv21 = conv_layer(conv20, shape=[3, 3, 1024, 1024], strides=[1, 1])
conv22 = conv_layer(conv21, shape=[3, 3, 1024, 1024], strides=[2, 2])
conv23 = conv_layer(conv22, shape=[3, 3, 1024, 1024], strides=[1, 1])
conv24 = conv_layer(conv23, shape=[3, 3, 1024, 1024], strides=[1, 1])

conv24_flat = tf.layers.flatten(conv24)

#fc1 = fc_layer(conv24_flat, size=4096)
#fc2 = fc_layer(fc1, size=1470)

#fc2_reshape = tf.reshape(fc2, [-1, 7, 7, 30])

fc1 = fc_layer(conv24_flat, size=4096)
#keep_prob = tf.placeholder(tf.float32)
keep_prob = 0.5
fc1_drop = tf.nn.dropout(fc1, keep_prob=keep_prob)

fc2 = fc_layer(fc1_drop, size=1470)
fc2_reshape = tf.reshape(fc2, [-1, 7, 7, 30])
### end important part

sess = tf.Session()
sess.run(tf.global_variables_initializer())

val_x = sess.run(feed_x)

result = sess.run(fc2_reshape, feed_dict={x: val_x})

print(result.shape)




'''
    Data playground
'''
# from gluoncv import data, utils
# from matplotlib import pyplot as plt

# train_dataset = data.VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
# val_dataset = data.VOCDetection(splits=[(2007, 'test')])
# print('Num of training images:', len(train_dataset))
# print('Num of validation images:', len(val_dataset))

# train_image, train_label = train_dataset[50]
# print('Image size (height, width, RGB):', train_image.shape)
# print('train label {}', train_label)

# class_ids = train_label[:, 4:5]
# bounding_boxes = train_label[:, :4]

# utils.viz.plot_bbox(train_image.asnumpy(), bounding_boxes, scores=None,
#                     labels=class_ids, class_names=train_dataset.classes)
# plt.show()
