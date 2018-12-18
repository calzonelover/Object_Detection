from model.custom_model import YOLOv1
from datasets.read import read_and_decode
import tensorflow as tf

BATCH_SIZE = 10
EPOCHS = 1

data_path = '/home/patomp/Desktop/YOLO/Object_Detection/datasets/train.tfrecords'  # address to save the hdf5 file

if __name__ == "__main__":
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=EPOCHS)
    image, annotation = read_and_decode(filename_queue, BATCH_SIZE)
    init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

    YOLO_model = YOLOv1(batch_size=BATCH_SIZE)

    sess = tf.Session()
    sess.run(init_op)
    YOLO_model.init_variables()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Let's read off 3 batches just for example
    for i in range(12000):
        img, anno = sess.run([image, annotation])
        YOLO_model.train(img, anno)

    coord.request_stop()
    coord.join(threads)

    # YOLO_model.log_summary()
    # load data and split train & validation set
    # check if model already exist or not
    # loop iteration
        # loop feed in batch size