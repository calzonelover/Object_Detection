import tensorflow as tf
import os
import abc

class BaseModel:
    def __init__(self, summary_dir="summary", save_dir="model_repo", log_dir="log_dir", model_name="model",save_checkpoint_time=0.1):
        # base
        self.save_checkpoint_time = save_checkpoint_time
        self.save_dir = save_dir
        self.model_name = model_name
        self.summary_dir = summary_dir
        self.log_dir = log_dir
        # tf
        self.x = tf.placeholder(tf.float32, shape=None)
        self.y = tf.placeholder(tf.float32, shape=None)
        # session and log
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.train_writer = tf.summary.FileWriter(os.path.join(os.path.join(self.summary_dir, self.model_name), self.log_dir), self.sess.graph)
        self.merged = tf.summary.merge_all()
        self.EP_log = 0
    # fundamental func
    def init_variables(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)  
    def save(self, dir, sess):
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=self.save_checkpoint_time)
        saver.save(self.sess, os.path.join(self.save_dir, self.model_name))
    def restore(self, dir, sess):
        saver = tf.train.Save()
        saver.restore(self.sess, os.path.join(self.save_dir, self.model_name))
    @abc.abstractmethod
    def inference(self, x):
        return
    @abc.abstractmethod
    def predict(self, x):
        return
    @abc.abstractmethod
    def train(self, x_batch, y_label_batch):
        pass
    # utility
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    def conv2d(self, x, W, stride=[1, 1]):
        return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding='SAME')
    def max_pool(self, x, kernel_size=[2,2], stride=[2,2]):
        return tf.nn.max_pool(x, ksize=[1, kernel_size[0], kernel_size[1], 1],
                                    strides=[1, stride[0], stride[1], 1], padding='SAME')
    def conv_layer(self, input, shape):
        W = self.weight_variable(shape)
        b = self.bias_variable([shape[3]])
        return tf.nn.relu(self.conv2d(input, W) + b)
    def fc_layer(self, input, size):
        in_size = int(input.get_shape()[1])
        W = self.weight_variable([in_size, size])
        b = self.bias_variable([size])
        return tf.add(tf.matmul(input, W) + b)
