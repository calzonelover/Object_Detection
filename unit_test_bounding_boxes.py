import tensorflow as tf
import numpy as np

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

S = 7
B = 2
C = 20
threshold = 0.6

feed_conv = tf.truncated_normal(shape=[2, S, S, 5*B+C], stddev=0.1)

### start important part
conv_out = tf.placeholder(tf.float32, shape=[2, S, S, 5*B+C], name='input_x')
# extract anchor
boxes, boxes_confidence, boxes_cls = tf.split(conv_out, [4*B, B, C], 3)
boxes = tf.reshape(boxes, [-1, S, S, B, 4])
boxes_confidence = tf.reshape(boxes_confidence, [-1, S, S, B, 1])
softmaxed_boxes_cls = tf.nn.softmax(boxes_cls)
# bounding boxes
anchor_cls_prob = tf.reshape(tf.tile(softmaxed_boxes_cls, [1,1,1,B]), [-1,S,S,B,C])
anchor_score = tf.multiply(boxes_confidence, anchor_cls_prob)
anchor_cls = tf.argmax(anchor_score, -1)
anchor_cls_score = tf.reduce_max(anchor_score, -1)
# filter
selected_anchor = tf.greater(anchor_cls_score, threshold)
scores = tf.boolean_mask(anchor_cls_score, selected_anchor)
anchors = tf.boolean_mask(boxes, selected_anchor)
classes = tf.boolean_mask(anchor_cls, selected_anchor)

### end important part

sess = tf.Session()
sess.run(tf.global_variables_initializer())

val_conv = sess.run(feed_conv)
result = sess.run(anchor_cls_prob, feed_dict={conv_out: val_conv})

print(result.shape)