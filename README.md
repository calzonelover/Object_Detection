# YOLO object detection

## Plan
* Prepare datasets 
    * Pascal VOC 2007 with 20 classes
    * The VOC 2006 with 10 classes
* Write neural network graph connection of CNN with no FC and activate to each grid (this moment we are using leakyRelu)
* Write loss function from scatch
* Train and eval model

## Log
### Datasets
* each images contain slighly different pixel size  [~350, ~500, 3]
    * then we have to resize image before feed into the model
* label datasets contain 5 elements = [x_min, y_min, x_max, y_max, class_number, ??]
