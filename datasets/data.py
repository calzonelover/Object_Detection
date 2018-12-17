'''
This may help: http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
'''

import tensorflow as tf
from gluoncv import data, utils
from matplotlib import pyplot as plt
import numpy as np
import cv2


train_dataset = data.VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
val_dataset = data.VOCDetection(splits=[(2007, 'test')])
print('Num of training images:', len(train_dataset))
print('Num of validation images:', len(val_dataset))

train_image, train_label = train_dataset[52]
print('Image size (height, width, RGB):', train_image.shape)
print('Train label:', train_label)

# tf_resize_img = tf.image.resize_images(
#     tf.reshape(train_image.asnumpy(), [1, train_image.shape[0], train_image.shape[1], train_image.shape[2]]),
#     size=[448, 448])

# print(tf_resize_img.shape)
# a = tf.reshape(tf_resize_img, [448, 448, 3])
# cv2.imwrite('out_img.jpg', a)

resized_img = cv2.resize(train_image.asnumpy(), (448, 448))
resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
train_image = cv2.cvtColor(train_image.asnumpy(), cv2.COLOR_BGR2RGB)
print(resized_img.shape)
cv2.imwrite('out_img.jpg', resized_img)
cv2.imwrite('ori.img.jpg', train_image)


# def load_data():
#       x_train = np.arrays()
#       y_train = []
#       x_test = []
#       y_test = []
#       for i in range(len(train_dataset)):
#             train_image, train_label = train_dataset[i]
#             x_train.append(train_image.asnumpy())
#             y_train.append(train_label)
#       for i in range(len(val_dataset)):
#             test_image, test_label = val_dataset[i]
#             x_test.append(test_image.asnumpy())
#             y_test.append(test_label)
#       print(x_train.shape, y_train.shape)
#       return (x_train, y_train), (x_test, y_test)

# _1, _2, _3, _4 = load_data()

'''
from gluoncv import data, utils
from matplotlib import pyplot as plt

train_dataset = data.VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
val_dataset = data.VOCDetection(splits=[(2007, 'test')])
print('Num of training images:', len(train_dataset))
print('Num of validation images:', len(val_dataset))

train_image, train_label = train_dataset[8]
print('Image size (height, width, RGB):', train_image.shape)

bounding_boxes = train_label[:, :4]
print('Num of objects:', bounding_boxes.shape[0])
print('Bounding boxes (num_boxes, x_min, y_min, x_max, y_max):\n',
      bounding_boxes)

class_ids = train_label[:, 4:5]
print('Class IDs (num_boxes, ):\n', class_ids)

utils.viz.plot_bbox(train_image.asnumpy(), bounding_boxes, scores=None,
                    labels=class_ids, class_names=train_dataset.classes)
plt.show()
'''