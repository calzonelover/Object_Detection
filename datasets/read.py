'''
https://www.programcreek.com/python/example/90442/tensorflow.VarLenFeature
http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/?fbclid=IwAR0HqMcZgVkZK-2M7emS7XFFSBnpV3wEXz4rtzLc-vijTE9k9p2rfYl97IY
'''

import tensorflow as tf


def read_and_decode(filename_queue, batch_size):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    feature = {'train/image': tf.FixedLenFeature([], tf.string),
                'train/label_xmin': tf.VarLenFeature(tf.float32),
                'train/label_ymin': tf.VarLenFeature(tf.float32),
                'train/label_xmax': tf.VarLenFeature(tf.float32),
                'train/label_ymax': tf.VarLenFeature(tf.float32),
                'train/label_class': tf.VarLenFeature(tf.float32),
                'train/label_hard': tf.VarLenFeature(tf.float32),
    }

    features = tf.parse_single_example(serialized_example, features=feature)

    image = tf.decode_raw(features['train/image'], tf.float32)
    

    # label = tf.sparse_tensor_to_dense(features['train/label_xmin'], default_value=0)
    # label_xmin = tf.cast(features['train/label_xmin'].values, tf.float32)
    # label = tf.reshape(label, [-1, 1])

    label = tf.stack([features['train/label_%s'%x].values for x in ['xmin', 'ymin', 'xmax', 'ymax', 'class', 'hard']])
    label = tf.transpose(label, [1, 0])
    # label = tf.cast(features['train/label_xmin'], tf.float32)

    image = tf.reshape(image, [448, 448, 3])

    print("Image dim:", image.get_shape())
    print("label dim: ", label.get_shape())

    # Creates batches by randomly shuffling tensors
    # images, labels_xmin = tf.train.slice_input_producer([image, label_xmin], shuffle=True)
    images, labels = tf.train.batch([image, label], batch_size=batch_size, capacity=32, num_threads=1, dynamic_pad=True)
    mask = tf.greater(tf.reduce_sum(labels, axis=-1), 1)
    return images, labels