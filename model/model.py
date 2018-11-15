import tensorflow as tf

from model.base_model import BaseModel

class YOLOv1(BaseModel):
    def __init__(self, S=7, n_bounding_boxes=5, n_classes=10, img_dim=[256, 256, 3]):
        super(YOLOv1, self).__init__()
        with self.graph.as_default():
            # paramater
            self.s = S
            self.bb = n_bounding_boxes
            self.n_classes = n_classes
            # graph connection of cnn net (out dim has to be S*S*bb*(5+n_classes) )
            # loss
            # optimizer
    def yolo_filter_boxes(self, box_confidence, boxes, box_class_probs, threshold = 0.6):
        pass
    def inference(self):
        pass
    def predict(self, cutoff_obj, threshold = 0.6):
        pass