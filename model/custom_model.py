import tensorflow as tf
import numpy as np

from model.base_model import BaseModel

class YOLOv1(BaseModel):
    def __init__(self, S=7, n_bounding_boxes=2, n_classes=20, input_dim=[448, 448, 3], batch_size=5, learning_rate=1e-3, threshold=0.6):
        super(YOLOv1, self).__init__()
        # parameters
        self.S = S
        self.B = n_bounding_boxes
        self.C = n_classes
        self.batch_size = batch_size
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.s_width = input_dim[0]/self.S
        self.s_height = input_dim[1]/self.S
        with self.graph.as_default():
            with tf.name_scope("YOLO_parameters"):
                self.lambda_coord = tf.constant(5.0, name='lambda_coord')
                self.lambda_noobj = tf.constant(0.5, name='lambda_noobj')
            # placeholder
            with tf.name_scope("placeholder"):
                self.x = tf.placeholder(tf.float32, shape=[self.batch_size, input_dim[0], input_dim[1], input_dim[2]], name="x_input")
                self.y_label = tf.placeholder(tf.float32, shape=[self.batch_size, None, 6], name="y_label")
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
                
                self.conv_out = tf.truncated_normal(shape=[self.batch_size, self.S, self.S, 5*self.B+self.C], stddev=0.1)
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
                self.boxes_shifted_x = tf.multiply(tf.concat([self.dummy_boxes_indices, self.dummy_boxes_indices], axis=2), self.s_width)
                self.boxes_shifted_y = tf.multiply(tf.concat([self.dummy_boxes_indices, self.dummy_boxes_indices], axis=2), self.s_height)
                self.boxes_global_x = tf.add(self.boxes_pixel_width, self.boxes_shifted_x)
                self.boxes_global_y = tf.add(self.boxes_pixel_height, self.boxes_shifted_y)
                self.boxes_global_xmax = tf.reshape(tf.add(self.boxes_global_x, tf.div(self.boxes_pixel_width, 2.0)), [-1, self.S,self.S,self.B,1 ])
                self.boxes_global_xmin = tf.reshape(tf.subtract(self.boxes_global_x, tf.div(self.boxes_pixel_width, 2.0)), [-1, self.S,self.S,self.B,1 ])
                self.boxes_global_ymax = tf.reshape(tf.add(self.boxes_global_y, tf.div(self.boxes_pixel_height, 2.0)), [-1, self.S,self.S,self.B,1 ])
                self.boxes_global_ymin = tf.reshape(tf.subtract(self.boxes_global_y, tf.div(self.boxes_pixel_height, 2.0)), [-1, self.S,self.S,self.B,1 ])
                self.boxes_global_position = tf.concat([self.boxes_global_xmin, self.boxes_global_xmax, self.boxes_global_ymin, self.boxes_global_ymax], 4)
                self.boxes_global_position_flat = tf.reshape(self.boxes_global_position, [-1, 1, self.S*self.S*self.B, 4])
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
                # print(self.anchor_cls.get_shape(), self.selected_anchor.get_shape(), self.classes.get_shape())
            with tf.name_scope("shred_y_label"):
                self.label_pos, self.label_cls, self.label_hard = tf.split(self.y_label, [4, 1, 1], 2)
                self.label_pos_tile = tf.tile( tf.reshape(self.label_pos, [self.batch_size, -1, 1, 4]), [1, 1, self.S*self.S*self.B, 1]) # dim [BS, ?, S*S*B, 4]
                self.label_xmin = tf.slice(self.label_pos, [0, 0, 0], [-1, -1, 1])
                self.label_xmax = tf.slice(self.label_pos, [0, 0, 2], [-1, -1, 1])
                self.label_ymin = tf.slice(self.label_pos, [0, 0, 1], [-1, -1, 1])
                self.label_ymax = tf.slice(self.label_pos, [0, 0, 3], [-1, -1, 1])
                self.label_global_x = tf.reduce_mean(tf.add(self.label_xmin, self.label_xmax), axis=-1)
                self.label_global_y = tf.reduce_mean(tf.add(self.label_ymin, self.label_ymax), axis=-1)
                self.r_label_width  = tf.sqrt(tf.subtract(self.label_xmax, self.label_xmin))
                self.r_label_height = tf.sqrt(tf.subtract(self.label_ymax, self.label_ymin))
            with tf.name_scope("non-max-suppression"):
                self.nms_indices = tf.image.non_max_suppression(self.anchors, self.scores, max_output_size=5, iou_threshold=0.5)
                self.nms_scores = tf.gather(self.scores, self.nms_indices)
                self.nms_boxes = tf.gather(self.anchors, self.nms_indices)
                self.nms_cls = tf.gather(self.classes, self.nms_indices)
            # loss
            with tf.name_scope("Loss"):
                with tf.name_scope("IOU"):
                    self.boxes_g_pos_flat_xmin, self.boxes_g_pos_flat_xmax, self.boxes_g_pos_flat_ymin, self.boxes_g_pos_flat_ymax = tf.split(self.boxes_global_position_flat, [1,1,1,1], axis=-1)
                    self.label_pos_tile_xmin, self.label_pos_tile_ymin, self.label_pos_tile_xmax, self.label_pos_tile_ymax = tf.split(self.label_pos_tile, [1,1,1,1], axis=-1)
                    self.x_left_down_corner = tf.maximum(self.boxes_g_pos_flat_xmin, self.label_pos_tile_xmin)
                    self.y_left_down_corner = tf.maximum(self.boxes_g_pos_flat_ymin, self.label_pos_tile_ymin)
                    self.x_right_up_corner = tf.minimum(self.boxes_g_pos_flat_xmax, self.label_pos_tile_xmax)
                    self.y_right_up_corner = tf.minimum(self.boxes_g_pos_flat_ymax, self.label_pos_tile_ymax)
                    self.intersect_area = tf.maximum(tf.add(tf.subtract(self.x_right_up_corner, self.x_left_down_corner), 1), 0)*tf.maximum(tf.add(tf.subtract(self.y_right_up_corner, self.y_left_down_corner), 1), 0)
                    self.label_area = tf.multiply(tf.add(tf.subtract(self.label_pos_tile_xmax, self.label_pos_tile_xmin), 1), tf.add(tf.subtract(self.label_pos_tile_ymax, self.label_pos_tile_ymin), 1))
                    self.pred_area = tf.multiply(tf.add(tf.subtract(self.boxes_g_pos_flat_xmax, self.boxes_g_pos_flat_xmin), 1), tf.add(tf.subtract(self.boxes_g_pos_flat_ymax, self.boxes_g_pos_flat_ymin), 1))
                    self.iou = tf.reshape(tf.div(self.intersect_area, tf.add(self.label_area, self.pred_area)), [self.batch_size, -1, self.S*self.S*self.B])
                    self.max_iou = tf.reshape(tf.reduce_max(self.iou, axis=-1), [self.batch_size, -1, 1])
                    self.I_obj = tf.to_float(tf.equal(self.iou, self.max_iou))
                    print("I_obj ", self.I_obj.get_shape())
                    self.I_noobj = tf.to_float(tf.greater(1e-10, self.iou))
                    print("I_noobj ",self.I_noobj.get_shape())
                    # print(self.iou.get_shape(), self.max_iou.get_shape(), self.I_obj.get_shape())
                    # print("IOU: {}, max_IOU dim: {}".format(self.iou.get_shape(), self.max_iou.get_shape()))
                with tf.name_scope("localization"):
                    self.boxes_global_x_rs = tf.reshape(self.boxes_global_x, [-1, 1, self.S*self.S*self.B])
                    self.resp_boxes_x = tf.reduce_max(tf.multiply(self.I_obj, self.boxes_global_x_rs), axis=-1) # [BS, ?]
                    self.boxes_global_y_rs = tf.reshape(self.boxes_global_y, [-1, 1, self.S*self.S*self.B])
                    self.resp_boxes_y = tf.reduce_max(tf.multiply(self.I_obj, self.boxes_global_y_rs), axis=-1) # [BS, ?]
                    self.loss_local_x2 = tf.square(tf.subtract(self.resp_boxes_x, self.label_global_x))
                    self.loss_local_y2 = tf.square(tf.subtract(self.resp_boxes_y, self.label_global_y))
                    self.loss_local = self.lambda_coord*tf.reduce_sum(tf.add(self.loss_local_x2, self.loss_local_y2)) # scalar
                with tf.name_scope("width"):
                    self.boxes_pixel_sqrt_width_rs = tf.reshape(self.boxes_pixel_sqrt_width, [self.batch_size, 1, self.S*self.S*self.B])
                    self.boxes_pixel_sqrt_height_rs = tf.reshape(self.boxes_pixel_sqrt_height, [self.batch_size, 1, self.S*self.S*self.B])
                    self.resp_boxes_sqrt_width = tf.reduce_max(tf.multiply(self.I_obj, self.boxes_pixel_sqrt_width_rs), axis=-1) # [BS, ?]
                    self.resp_boxes_sqrt_height = tf.reduce_max(tf.multiply(self.I_obj, self.boxes_pixel_sqrt_height_rs), axis=-1) # [BS, ?]
                    self.r_label_width_rs = tf.reshape(self.r_label_width, [self.batch_size, -1])
                    self.r_label_height_rs = tf.reshape(self.r_label_height, [self.batch_size, -1])
                    self.loss_width_w = tf.square(tf.subtract(self.resp_boxes_sqrt_width, self.r_label_width_rs), name='loss_width_horizontal')
                    self.loss_width_h = tf.square(tf.subtract(self.resp_boxes_sqrt_height, self.r_label_height_rs), name='loss_width_vertical')
                    self.loss_width = tf.multiply(self.lambda_coord, tf.reduce_sum(tf.add(self.loss_width_w, self.loss_width_h)), name='loss_width')
                with tf.name_scope("confidence"):
                    self.boxes_confidence_rs = tf.reshape(self.boxes_confidence, [-1, 1, self.S*self.S*self.B])
                    self.max_iou_rs = tf.reshape(self.max_iou, [self.batch_size, -1])
                    self.resp_boxes_confidence = tf.reduce_max(tf.multiply(self.I_obj, self.boxes_confidence_rs), axis=-1)
                    self.noobj_boxes_confidence = tf.multiply(self.I_noobj, self.boxes_confidence_rs) # [BS, -1, S*S*B]
                    self.loss_conf_obj = tf.reduce_sum(tf.square(tf.subtract(self.resp_boxes_confidence, self.max_iou_rs)), name='loss_confidence_obj') # scalar
                    self.loss_conf_noobj = tf.reduce_sum(tf.square(tf.subtract(self.noobj_boxes_confidence, self.iou)), name='loss_confidence_noobj') # scalar
                    self.loss_conf = tf.add(self.loss_conf_obj, self.loss_conf_noobj, name='loss_confidence')
                with tf.name_scope("clssification"):
                    self.loss_cls = None
                # self.total_loss = tf.div(self.loss_local + self.loss_width + self.loss_conf + self.loss_cls, self.batch_size, name='total_loss')
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
    def log_summary(self): # for log per EP only
        with self.graph.as_default():
            x0 = np.random.randint(10, size=(5, 448, 448, 3))
            y0_label = np.random.randint(10, size=(5, 3, 6))
            summary = self.sess.run([self.merged], feed_dict={self.x: x0, self.y_label: y0_label})
            self.train_writer.add_summary(summary, 1)