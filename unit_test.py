# '''
#     For tensorflow dimensional testing
#     reference for check dimension :  https://hackernoon.com/understanding-yolo-f5a74bbc7967
# '''
# import tensorflow as tf

# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
# def conv2d(x, W, strides):
#     return tf.nn.conv2d(x, W, strides=[1, strides[0], strides[1], 1], padding='SAME')
# def max_pool(x, kernel_size=[2,2], strides=[2,2]):
#     return tf.nn.max_pool(x, ksize=[1, kernel_size[0], kernel_size[1], 1],
#                                 strides=[1, strides[0], strides[1], 1], padding='SAME')
# def conv_layer(input, shape, strides=[1,1]):
#     W = weight_variable(shape)
#     b = weight_variable([shape[3]])
#     return tf.nn.leaky_relu(tf.add(conv2d(input, W, strides=strides), b))
# def fc_layer(input, size):
#     in_size = int(input.get_shape()[1])
#     W = weight_variable([in_size, size])
#     b = weight_variable([size])
#     return tf.add(tf.matmul(input, W), b)


# feed_x = tf.truncated_normal(shape=[2, 448, 448, 3], stddev=0.1)

# ### start important part
# x = tf.placeholder(tf.float32, shape=[2, 448, 448, 3], name='input_x')
# conv1 = conv_layer(x, shape=[7, 7, 3, 64], strides=[2,2])
# max_pool1 = max_pool(conv1, kernel_size=[2,2], strides=[2,2])
# conv2 = conv_layer(max_pool1, shape=[3, 2, 64, 192], strides=[1,1])
# max_pool2 = max_pool(conv2, kernel_size=[2,2], strides=[2,2]) # would have dim (2, 56, 56, 192) as example in reference
# ### end important part

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# val_x = sess.run(feed_x)
# result = sess.run(max_pool2, feed_dict={x: val_x})

# print(result.shape)




'''
    Data playground
'''
from gluoncv import data, utils
from matplotlib import pyplot as plt

train_dataset = data.VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
val_dataset = data.VOCDetection(splits=[(2007, 'test')])
print('Num of training images:', len(train_dataset))
print('Num of validation images:', len(val_dataset))

train_image, train_label = train_dataset[50]
print('Image size (height, width, RGB):', train_image.shape)
print('train label', train_label, 'shape', train_label.shape)


train_image, train_label = train_dataset[52]
print('Image size (height, width, RGB):', train_image.shape)
print('train label', train_label, 'shape', train_label.shape)


train_image, train_label = train_dataset[100]
print('Image size (height, width, RGB):', train_image.shape)
print('train label', train_label, 'shape', train_label.shape)
# class_ids = train_label[:, 4:5]
# bounding_boxes = train_label[:, :4]

# utils.viz.plot_bbox(train_image.asnumpy(), bounding_boxes, scores=None,
#                     labels=class_ids, class_names=train_dataset.classes)
# plt.show()