'''
This may help: http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
'''
import tensorflow as tf
from gluoncv import data, utils
import numpy as np
import cv2

train_filename = 'train.tfrecords'
val_filename = 'val.tfrecords'
size_img = (448, 448)

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# def _int64_feature(value):
#     value = value if type(value) == list else [value]
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
# def _bytes_feature(value):
#     value = value if type(value) == list else [value]
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
# def _float_feature(value):
#     value = value if type(value) == list else [value]
#     return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def resize_img(_img):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    _img = cv2.resize(_img.asnumpy(), size_img)
    _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
    _img = _img.astype(np.float32)
    return _img

# Write TFrecord file (training)
writer = tf.python_io.TFRecordWriter(train_filename)

# read whole dataset
train_dataset = data.VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])

for i in range(len(train_dataset)):
    image, label = train_dataset[i]
    image = resize_img(image)
    feature = {
        'train/label_xmin': _float_feature(label[:,0]),
        'train/label_ymin': _float_feature(label[:,1]),
        'train/label_xmax': _float_feature(label[:,2]),
        'train/label_ymax': _float_feature(label[:,3]),
        'train/label_class': _float_feature(label[:,4]),
        'train/label_hard': _float_feature(label[:,5]),
        'train/image': _bytes_feature(tf.compat.as_bytes(image.tostring()))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

writer.close()

# Write TFrecord file (test)
# writer = tf.python_io.TFRecordWriter(val_filename)
# val_dataset = data.VOCDetection(splits=[(2007, 'test')])
# for i in range(len(val_dataset)):
#     image, label = val_dataset[i]
#     image = resize_img(image)
#     feature = {
#         'val/label_xmin': _float_feature(label[:,0]),
#         'val/label_ymin': _float_feature(label[:,1]),
#         'val/label_xmax': _float_feature(label[:,2]),
#         'val/label_ymax': _float_feature(label[:,3]),
#         'val/label_class': _float_feature(label[:,4]),
#         'val/label_hard': _float_feature(label[:,5]),
#         'val/image': _bytes_feature(tf.compat.as_bytes(image.tostring()))
#     }
#     example = tf.train.Example(features=tf.train.Features(feature=feature))
#     writer.write(example.SerializeToString())

# writer.close()
