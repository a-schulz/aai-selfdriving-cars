# aai-selfdriving-cars
### Content
1.  [before Implementation](#before-implementation)
2.  [Aim](#aim)
3.  [The Architecture](#the-architecture)
4.  [Repository structure](#repository-structure)
5. [Why not only one AI Model](#why-not-only-one-ai-model)

## before Implementation
Before we could start our implementation, we tried to get a feeling of the problem space.
In order to do that, we collect possible states of a car, observation from sensors a car can collect and the action a driver can do.  We used that finally in a rule base.
This pre work is in /documents/rule_base.pdf stored.



## Aim
Because a self driving car is a very complex system, we implement just a proof of concept for object detection and classification from a camera sensor.
To achieve this, we combine different AI models:

* YOLOv8 
  * used to detect traffic lights and other objects
  * also used to generate dataset for traffic light classification
* [Roboflow Model](https://universe.roboflow.com/tu-wien-pfowz/traffic-sign-detection-yolov8/model/3)
    * detect signs in a pure image. (Yolo detects only stop signs, so we needed another detection model)
* Self trained model to detect the type of the traffic sign (ResNet50)
    *  This model uses the image from the Roboflow model, the image will be cropped and the cropped image will be classified by the ResNet50 model
* Traffic light classification (from lecture but with different model)
For specific objects like traffic lights and traffic signs we use the detected position of these to classify the exact class of this category.


## usage
First of all install the required packages with
```
cd src
pip install -r requirements.txt
```

Than set up all environment variables in src/.env:
```
# https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
TRAFFIC_SIGN_DIR=~/kaggle/gtsrb-german-traffic-sign
DATASET_DIR=~/aai-selfdriving-cars/dataset
TRAFFIC_LIGHT_ORIGINAL_DATA=~/aai-selfdriving-cars/dataset/traffic_light/original_data/
TRAFFIC_LIGHT_CUSTOM_DATA=~/aai-selfdriving-cars/dataset/traffic_light/custom_data/
TRAFFIC_LIGHT_DROP_DATA=~/aai-selfdriving-cars/dataset/traffic_light/drop_outs/
TRAFFIC_SIGN_DIR=~/aai-selfdriving-cars/dataset/traffic_sign/
```
> To try the main file with own pictures please store them in:
> dataset/test_data

### usage of the notebooks
The precised usage descriptions are in the specific notebooks.
Please have a look to each of them.

To detect Objects in images you should use the "main" notebook.
In this notebook are all the models combined.
So it can detect all yolov8 pretrained objects, traffic signs and lights.

To build the traffic light classificator and to create its train dataset have a look in "traffic_light_classification.py"

The ResNet notebook defines all required Objects and helper to train and use the traffic sign classification.

## Repository structure
Here is a list of the directories and they content

### dataset
Here is the data for training the traffic sing and light as well to test images.

#### traffic light
Contains original dataset, dropped data and the actual custom dataset where the model is trained on.

### src
Here all the source, except the data to run the classification

### documents
Here are the documents we create before we implement the ai system.