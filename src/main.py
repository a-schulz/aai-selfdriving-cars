# Zum starten des Models
# zeigt ein Bild an, welches durch das Model gezogen wird.
# Anzeige, welche objekte erkannt wurden. Mit bounding box
# bounding box wird extrahiert und an klassifikator weitergegeben fuer die genaue klassifikation
# ergebniss davon ausgeben
# print action (drive left, drive right, ...)

from ultralytics import YOLO
import cv2
import numpy as np

# get images from directory
import os
def is_image(filename):
    image_extensions = ['.jpg', '.jpeg', '.png']  
    _, file_extension = os.path.splitext(filename)
    return file_extension.lower() in image_extensions

def get_images(dir):
    # load images which have to be evaluated from the ai into a list
    images = []
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and is_image(file):
            # images contains images in RGB format
            images.append(cv2.imread(path))
    return images

# TODO: path should be not hard coded
images = get_images('./dataset/test') # python script have to be run in ./aai-selfdriving-cars

## Detect content:
# Load a model
model = YOLO('yolov8m.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
# results get saved in runs/detect folder 
results = model.predict(images[0], save=True, save_txt = True)  # return a list of Results objects

# Process results list
for result in results:
    print(result)

#    result = result.numpy() # to get an numpy object not a tensor
#    boxes = result.boxes.xyxy # Boxes object for bbox outputs
#    print(boxes)
#    print(boxes[0])
    #masks = result.masks  # Masks object for segmentation masks outputs
    #keypoints = result.keypoints  # Keypoints object for pose outputs
    #probs = result.probs  # Class probabilities for classification outputs
    
    #print(f'boxes: {boxes}'.format(boxes))


exit()

# plot the result:
def draw_rectangle(image, points, color = (0, 255, 0), thick = 2):
    if len(points) != 4:
        raise ValueError("Die Anzahl der Eckpunkte muss 4 sein.")
    
    x1, y1, x2, y2 = points

    # Zeichne das Rechteck auf das Bild
    cv2.rectangle(image, (x1,y1), (x2,y2), color=color, thickness=thick)

while True: 
    draw_rectangle(images[0], boxes[0])
    cv2.imshow("Image", images[0])

    if cv2.waitKey(1) == 27:
        break

input()