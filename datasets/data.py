'''
import tensorflow as tf
from gluoncv import data, utils
from matplotlib import pyplot as plt
import cv2

train_dataset = data.VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
val_dataset = data.VOCDetection(splits=[(2007, 'test')])
print('Num of training images:', len(train_dataset))
print('Num of validation images:', len(val_dataset))

train_image, train_label = train_dataset[20]
print('Image size (height, width, RGB):', train_image.shape)
print('Train label:', train_label)





resized_img = cv2.resize(train_image.asnumpy(), (448, 448))
resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
train_image = cv2.cvtColor(train_image.asnumpy(), cv2.COLOR_BGR2RGB)
print(resized_img.shape)
cv2.imwrite('out_img.jpg', resized_img)
cv2.imwrite('ori.img.jpg', train_image)
'''

import tensorflow as tf

dataset = tf.data.TFRecordDataset('train.tfrecords')
BATH_SIZE = 64

def parser(serialized_example):
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
                'train/label_xmin': tf.VarLenFeature(tf.float32),
                'train/label_ymin': tf.VarLenFeature(tf.float32),
                'train/label_xmax': tf.VarLenFeature(tf.float32),
                'train/label_ymax': tf.VarLenFeature(tf.float32)
    }

    features = tf.parse_single_example(serialized_example, features=feature)

    image = tf.decode_raw(features['train/image'], tf.float32)
    

    # label = tf.sparse_tensor_to_dense(features['train/label_xmin'], default_value=0)
    label = tf.cast(features['train/label_xmin'].values, tf.float32)
    label = tf.reshape(label, [-1, 1])
    image = tf.reshape(image, [448, 448, 3])
    return image, label


dataset = dataset.map(parser)