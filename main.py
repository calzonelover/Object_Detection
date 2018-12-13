from model.custom_model import YOLOv1

if __name__ == "__main__":
    YOLO_model = YOLOv1()
    YOLO_model.init_variables()
    # YOLO_model.log_summary()
    # load data and split train & validation set
    # check if model already exist or not
    # loop iteration
        # loop feed in batch size