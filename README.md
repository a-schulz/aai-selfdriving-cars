# aai-selfdriving-cars
### Content
1.  [How we got here](#How-we-got-here)
2.  [Aim](#aim)
3.  [The Architecture](#the-architecture)
4.  [Repository structure](#repository-structure)
5. [Why not only one AI Model](#why-not-only-one-ai-model)

# How we got here
### First steps
Before we could start our implementation, we tried to get a feeling of the problem space.
In order to do that, we had to think of a human based solution, we collect possible states of a car, observation from sensors a car can collect and the action a driver can do.  We used that finally in a rule base.
Those represented the observations of the self-driving car. The surrounding and the actions for the vehicle. You can find those here: [rule_base](documents/rule_base.adoc) and [state_observation](documents/state_observation_action.pdf)
### Next steps
Now we needed to think of a useful architecture. We decided to use multiple optimized models for specific tasks such as traffic light classification and traffic sign classification.
### Implementation
Most of the implementation we tried new things had to fight with hardware problems like the difference of cpu cuda and out of memory errors. But in the end we got a working solution with multiple models. On the way there we search for datasets and read code of other users mostly on kaggle.



## Aim
Because a self driving car is a very complex system, we implement just a proof of concept for object detection and classification from a camera sensor.
To achieve this, we combine different AI models:

* YOLOv8 
  * used to detect traffic lights and other objects
  * also used to generate dataset for traffic light classification
* [Roboflow Model](https://universe.roboflow.com/tu-wien-pfowz/traffic-sign-detection-yolov8/model/3)
    * detect signs in a pure image. (Yolo detects only stop signs, so we needed another detection model)
    * using the yolov8 Detection API
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

### To run the code you have to set the following environment variables
I have set them using .env file in src folder.
See [demo.env](src/demo.env) for an example.
The most important one is the path to the dataset folder. `DATASET_DIR`

The heart of our project is the [main.ipynb](src/main.ipynb) file. Here you find the definitions of all the models.
Respectively we have separated some code into other files. Otherwise the main file would be too big.

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

# Our problems
## the dataset
## the extracted image form original road scene
* traffic lights detected from yolo may have no light, e.g. if the traffic sign shows away from the camera
  the classificator is then every time wrong because of the training data set

* the image is very small so we added a summand corresponding to the boxing size