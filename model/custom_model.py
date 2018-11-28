import tensorflow as tf

from model.base_model import BaseModel

class YOLOv1(BaseModel):
    def __init__(self, S=7, n_bounding_boxes=2, n_classes=20, input_dim=[448, 448, 3], learning_rate=1e-3, threshold=0.6):
        super(YOLOv1, self).__init__()
        # parameters
        self.S = S
        self.B = n_bounding_boxes
        self.C = n_classes
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.s_width = input_dim[0]/self.S
        self.s_height = input_dim[1]/self.S
        with self.graph.as_default():
            # placeholder
            with tf.name_scope("placeholder"):
                self.x = tf.placeholder(tf.float32, shape=[None, input_dim[0], input_dim[1], input_dim[2]], name="x_input")
                self.y_label = tf.placeholder(tf.float32, shape=[None, 6], name="y_label")
            # graph connection of cnn net ( out dim has to be S*S*bb*(5+n_classes) )
            with tf.name_scope("Convolution"):
                self.conv1 = self.conv_layer(self.x, shape=[self.S, self.S, 3, 64], strides=[2, 2])	# shape = [size, size, old_channel, new_channel]
                self.max_pool1 = self.max_pool(self.conv1, kernel_size=[2, 2], strides=[2, 2])

                self.conv2 =  self.conv_layer(self.max_pool1, shape=[3, 3, 64, 192], strides=[1, 1])
                self.max_pool2 = self.max_pool(self.conv2, kernel_size=[2, 2], strides=[2, 2])

                self.conv3 = self.conv_layer(self.max_pool2, shape=[1, 1, 192, 128], strides=[1, 1])
                self.conv4 = self.conv_layer(self.conv3, shape=[3, 3, 128, 256], strides=[1, 1])
                self.conv5 = self.conv_layer(self.conv4, shape=[1, 1, 256, 256], strides=[1, 1])
                self.conv6 = self.conv_layer(self.conv5, shape=[1, 1, 256, 512], strides=[1, 1])

                self.max_pool3 = self.max_pool(self.conv6, kernel_size=[2, 2], strides=[2, 2])

                self.conv7 = self.conv_layer(self.max_pool3, shape=[1, 1, 512, 256], strides=[1, 1])
                self.conv8 = self.conv_layer(self.conv7, shape=[3, 3, 256, 512], strides=[1, 1])
                self.conv9 = self.conv_layer(self.conv8, shape=[1, 1, 512, 256], strides=[1, 1])
                self.conv10 = self.conv_layer(self.conv9, shape=[3, 3, 256, 512], strides=[1, 1])
                self.conv11 = self.conv_layer(self.conv10, shape=[1, 1, 512, 256], strides=[1, 1])
                self.conv12 = self.conv_layer(self.conv11, shape=[3, 3, 256, 512], strides=[1, 1])
                self.conv13 = self.conv_layer(self.conv12, shape=[1, 1, 512, 256], strides=[1, 1])
                self.conv14 = self.conv_layer(self.conv13, shape=[3, 3, 256, 512], strides=[1, 1])
                self.conv15 = self.conv_layer(self.conv14, shape=[1, 1, 512, 512], strides=[1, 1])
                self.conv16 = self.conv_layer(self.conv15, shape=[3, 3, 512, 1024], strides=[1, 1])

                self.max_pool4 = self.max_pool(self.conv16, kernel_size=[2, 2], strides=[2, 2])

                self.conv17 = self.conv_layer(self.max_pool4, shape=[1, 1, 1024, 512], strides=[1, 1])
                self.conv18 = self.conv_layer(self.conv17, shape=[3, 3, 512, 1024], strides=[1, 1])
                self.conv19 = self.conv_layer(self.conv18, shape=[1, 1, 1024, 512], strides=[1, 1])
                self.conv20 = self.conv_layer(self.conv19, shape=[3, 3, 512, 1024], strides=[1, 1])
                self.conv21 = self.conv_layer(self.conv20, shape=[3, 3, 1024, 1024], strides=[1, 1])
                self.conv22 = self.conv_layer(self.conv21, shape=[3, 3, 1024, 1024], strides=[2, 2])
                self.conv23 = self.conv_layer(self.conv22, shape=[3, 3, 1024, 1024], strides=[1, 1])
                self.conv24 = self.conv_layer(self.conv23, shape=[3, 3, 1024, 1024], strides=[1, 1])

                self.conv24_flat = tf.layers.flatten(self.conv24)

                self.fc1 = self.fc_layer(self.conv24_flat, size=4096)
                self.fc1_drop = tf.nn.dropout(self.fc1, keep_prob=1.0-0.5)

                self.fc2 = self.fc_layer(self.fc1_drop, size=1470)
                self.fc2_reshape = tf.reshape(self.fc2, [-1, self.S, self.S, 5*self.B+self.C])
                
                self.conv_out = tf.truncated_normal(shape=[2, self.S, self.S, 5*self.B+self.C], stddev=0.1)
            # bonding boxes and filtering
            with tf.name_scope("Bounding_Boxes"): 
                # size boxes = [-1, S, S, B, 4], size boxes_confidence = [-1, S, S, 1], size boxes_cls = [-1, S, S, 20]
                self.boxes, self.boxes_confidence, self.boxes_cls = tf.split(self.conv_out, [4*self.B, self.B, self.C], 3)
                self.boxes = tf.reshape(self.boxes, [-1, self.S, self.S, self.B, 4])
                self.boxes_confidence = tf.reshape(self.boxes_confidence, [-1, self.S, self.S, self.B, 1])
                self.softmaxed_boxes_cls = tf.nn.softmax(self.boxes_cls)
            with tf.name_scope("Position_Boxes"):
                self.boxes_local_x, self.boxes_local_y, self.boxes_local_w, self.boxes_local_h = tf.split(self.boxes, [1, 1, 1, 1], 4)
                self.boxes_pixel_width = tf.reshape(tf.multiply(self.boxes_local_w, self.s_width), [-1,self.S,self.S,self.B])
                self.boxes_pixel_height = tf.reshape(tf.multiply(self.boxes_local_h, self.s_height), [-1,self.S,self.S,self.B])
                self.boxes_pixel_sqrt_width = tf.sqrt(tf.multiply(self.boxes_local_w, self.s_width))
                self.boxes_pixel_sqrt_height = tf.sqrt(tf.multiply(self.boxes_local_h, self.s_height))
                self.dummy_boxes_indices = tf.reshape(tf.tile(tf.constant([[i for i in range(self.S)]],  dtype=tf.float32), [7,1]), [7,7,1])
                # tf.multiply(self.dummy_boxes_indices, self.s_width))
                self.boxes_shifted_x = tf.multiply(tf.concat([self.dummy_boxes_indices, self.dummy_boxes_indices], axis=2), self.s_width)
                self.boxes_shifted_y = tf.multiply(tf.concat([self.dummy_boxes_indices, self.dummy_boxes_indices], axis=2), self.s_height)
                self.boxes_global_x = tf.add(self.boxes_pixel_width, self.boxes_shifted_x)
                self.boxes_global_y = tf.add(self.boxes_pixel_height, self.boxes_shifted_y)

                self.boxes_global_xmax = tf.reshape(tf.add(self.boxes_global_x, tf.div(self.boxes_pixel_width, 2.0)), [-1, self.S,self.S,self.B,1 ])
                self.boxes_global_xmin = tf.reshape(tf.subtract(self.boxes_global_x, tf.div(self.boxes_pixel_width, 2.0)), [-1, self.S,self.S,self.B,1 ])
                self.boxes_global_ymax = tf.reshape(tf.add(self.boxes_global_y, tf.div(self.boxes_pixel_height, 2.0)), [-1, self.S,self.S,self.B,1 ])
                self.boxes_global_ymin = tf.reshape(tf.subtract(self.boxes_global_y, tf.div(self.boxes_pixel_height, 2.0)), [-1, self.S,self.S,self.B,1 ])
                self.boxes_global_position = tf.concat([self.boxes_global_xmin, self.boxes_global_xmax, self.boxes_global_ymin, self.boxes_global_ymax], 4)
            with tf.name_scope("Anchor"):
                self.anchor_cls_prob = tf.reshape(tf.tile(self.softmaxed_boxes_cls, [1,1,1,self.B]), [-1,self.S,self.S,self.B,self.C])
                self.anchor_score = tf.multiply(self.boxes_confidence, self.anchor_cls_prob)
                self.anchor_cls = tf.argmax(self.anchor_score, -1)
                self.anchor_cls_score = tf.reduce_max(self.anchor_score, -1)
            with tf.name_scope("Filter_Threshold"):
                self.selected_anchor = tf.greater(self.anchor_cls_score, self.threshold)
                self.scores = tf.boolean_mask(self.anchor_cls_score, self.selected_anchor)
                self.anchors = tf.boolean_mask(self.boxes_global_position, self.selected_anchor)
                self.classes = tf.boolean_mask(self.anchor_cls, self.selected_anchor)
            # loss
            with tf.name_scope("Loss"):
                with tf.name_scope("shred_y_label"):
                    self.label_pos, self.label_cls, self.label_hard = tf.split(self.y_label, [4, 1, 1], 1)
                    self.label_xmin = tf.slice(self.label_pos, [0, 0], [-1, 1])
                    self.label_xmax = tf.slice(self.label_pos, [0, 2], [-1, 1])
                    self.label_ymin = tf.slice(self.label_pos, [0, 1], [-1, 1])
                    self.label_ymax = tf.slice(self.label_pos, [0, 3], [-1, 1])
                    self.label_global_x = tf.reduce_mean(tf.add(self.label_xmin, self.label_xmax))
                    self.label_global_y = tf.reduce_mean(tf.add(self.label_ymin, self.label_ymax))
                    self.r_label_width  = tf.sqrt(tf.subtract(self.label_xmax, self.label_xmin))
                    self.r_label_height = tf.sqrt(tf.subtract(self.label_ymax, self.label_ymin))
                    # print(self.r_label_width.get_shape())
                with tf.name_scope("Filtered_Responsible"):
                    print(self.boxes_global_position.get_shape())
                    self.masked_responsible = None
                with tf.name_scope("non-max-suppression"):
                    self.nms_indices = tf.image.non_max_suppression(self.anchors, self.scores, max_output_size=5, iou_threshold=0.5)
                    self.nms_scores = tf.gather(self.scores, self.nms_indices)
                    self.nms_boxes = tf.gather(self.anchors, self.nms_indices)
                    self.nms_cls = tf.gather(self.classes, self.nms_indices)
                with tf.name_scope("localization"):
                    self.loss_local = None
                with tf.name_scope("width"):
                    self.loss_width = None
                with tf.name_scope("confidence"):
                    self.loss_conf = None
                with tf.name_scope("clssification"):
                    self.loss_cls = None
                self.loss = None
            # optimizer
            with tf.name_scope("Optimizer"):
                self.optimizer = None # tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            with tf.name_scope("Goodness"):
                self.accuracy = None
    def inference(self, **kwargs):
        scores, boxes, classes = self.sess.run([self.nms_scores, self.nms_boxes, self.nms_cls], feed_dict={})
        return scores, boxes, classes
    def predict(self, **kwargs):
        return
    def train(self, x_batch, y_label_batch):
        pass