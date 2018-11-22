import tensorflow as tf

from model.base_model import BaseModel

class YOLOv1(BaseModel):
    def __init__(self, S=7, n_bounding_boxes=2, n_classes=20, input_dim=[364, 480, 3], learning_rate=1e-3, threshold=0.6):
        super(YOLOv1, self).__init__()
        # parameters
        self.S = S
        self.B = n_bounding_boxes
        self.C = n_classes
        self.threshold = threshold 
        self.learning_rate = learning_rate
        with self.graph.as_default():
            # placeholder
            with tf.name_scope("placeholder"):
                self.x = tf.placeholder(tf.float32, shape=[None, input_dim[0], input_dim[1], input_dim[2]], name="x_input")
                self.y_label = tf.placeholder(tf.float32, shape=[None, 6], name="y_label")
            # graph connection of cnn net ( out dim has to be S*S*bb*(5+n_classes) )
            with tf.name_scope("Convolution"):
                # suppose this done by Poom
                self.conv_out = tf.truncated_normal(shape=[2, self.S, self.S, 5*self.B+self.C], stddev=0.1)
            # bonding boxes and filtering
            with tf.name_scope("Bounding_Boxes"):
                self.boxes, self.boxes_confidence, self.boxes_cls = tf.split(self.conv_out, [4*self.B, self.B, self.C], 3)
                self.boxes = tf.reshape(self.boxes, [-1, self.S, self.S, self.B, 4])
                self.boxes_confidence = tf.reshape(self.boxes_confidence, [-1, self.S, self.S, self.B, 1])
                self.softmaxed_boxes_cls = tf.nn.softmax(self.boxes_cls)
            with tf.name_scope("Anchor"):
                self.anchor_cls_prob = tf.reshape(tf.tile(self.softmaxed_boxes_cls, [1,1,1,self.B]), [-1,self.S,self.S,self.B,self.C])
                self.anchor_score = tf.multiply(self.boxes_confidence, self.anchor_cls_prob)
                self.anchor_cls = tf.argmax(self.anchor_score, -1)
                self.anchor_cls_score = tf.reduce_max(self.anchor_score, -1)
            with tf.name_scope("Filter_Threshold"):
                self.selected_anchor = tf.greater(self.anchor_cls_score, self.threshold)
                self.scores = tf.boolean_mask(self.anchor_cls_score, self.selected_anchor)
                self.anchors = tf.boolean_mask(self.boxes, self.selected_anchor)
                self.classes = tf.boolean_mask(self.anchor_cls, self.selected_anchor)
            # loss
            with tf.name_scope("Loss"):
                with tf.name_scope("shred_y_label"):
                    self.label_pos, self.label_cls, self.label_hard = tf.split(self.y_label, [4, 1, 1], 1)       
                    self.label_xmin = tf.slice(self.label_pos, [0, 0], [-1, 1])
                    self.label_xmax = tf.slice(self.label_pos, [0, 2], [-1, 1])
                    self.label_ymin = tf.slice(self.label_pos, [0, 1], [-1, 1])
                    self.label_ymax = tf.slice(self.label_pos, [0, 3], [-1, 1])
                    self.r_label_width  = tf.sqrt(tf.subtract(self.label_xmax, self.label_xmin)) 
                    self.r_label_height  = tf.sqrt(tf.subtract(self.label_ymax, self.label_ymin)) 
                    print(self.r_label_width.get_shape())
                with tf.name_scope("non-max-suppression"):
                    self.nms_indices = tf.image.non_max_suppression(self.anchors, self.scores, max_output_size=5, iou_threshold=0.5)
                    self.nms_scores = tf.gather(self.scores, self.nms_indices)
                    self.nms_boxes = tf.gather(self.anchors, self.nms_indices)
                    self.nms_cls = tf.gather(self.classes, self.nms_indices)
                with tf.name_scope("localization"):
                    self.loss_local = None
                with tf.name_scope("width"):
                    self.s_width = input_dim[1]/self.S
                    self.s_height = input_dim[2]/self.S
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