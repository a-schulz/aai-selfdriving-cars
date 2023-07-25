# aai-selfdriving-cars
The goal of this Project  is to create a rudimentary AI System to identify different street situations form a driver point of view.
To achieve it, we analyze a picture with the yolov8n model to approximate a realtime application (disclaimer: it is not a realtime system).
For specific objects like traffic lights and traffic signs we use the detected position of these to classify the exact class of this category.

## The Architecture 

## Dataset 
To fine tune the yolov8n model we download training pictures with the src/download_images.py script.
After that we label this pictures with the website https://www.makesense.ai/


## Information about Data Sets
Here are some use full Information about creating own datasets.
We can use these information to create a data set for our own purpose.
The structure of the dataset, described here, amies a uncomplicated way to train YOLOv8 models.

# The YOLO v8
Yolo (You only look once) is an Deep Learning Model for computer vision tasks (e.g object detection, segmentation)

## YOLO v8 pretrained  models
There are 6 pretrained models available which just differs in size (of Parameters) and runtime.
They got trained at the COCO dataset and cover 79 categories:
0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 
10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 
20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 
50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 
60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 
70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'

### Folder structure

./"Name" dataset
    images/
        test
        train
        valid

### Data Tips and Tricks
* 20% valid - 80% train data or 30% - 70% 
* ensure a good diversity and variance (no sequences a video clip as frames, because there are heavily correlated)
* to care of luminosity and different angles much more data have to be used
* no overlapping between training and validation data
* amount of data for different classes should be balanced (200 car pictures, 200 truck pictures)
