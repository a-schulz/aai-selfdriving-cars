# Zum starten des Models
# zeigt ein Bild an, welches durch das Model gezogen wird.
# Anzeige, welche objekte erkannt wurden. Mit bounding box
# bounding box wird extrahiert und an klassifikator weitergegeben fuer die genaue klassifikation
# ergebniss davon ausgeben
# print action (drive left, drive right, ...)

from custom_utils import get_images
from ultralytics import YOLO
import cv2
import numpy as np
#import tensorflow as tf
# get images from directory
images = get_images('./dataset/test') # python script have to be run in ./aai-selfdriving-cars

img = images
print(images)
"""
# TODO: path should be not hard coded
path = "./dataset/test/traffic_light.jpg"
width = 640
hight = 640
img = cv2.imread(path)
img = cv2.resize(img, (width, hight))
"""

## Detect content:
# Load a model
model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
# results get saved in runs/detect folder 
results = model.predict(img, save=True, save_txt = True)  # return a list of Results objects

# Process results list
for result in results:
    print(result.boxes.xyxy)

# plot the result:
def draw_rectangle(image, points, color = (0, 255, 0), thick = 2):
    if len(points) != 4:
        raise ValueError("Die Anzahl der Eckpunkte muss 4 sein.")
    
    for i, p in enumerate(points):
        if i % 2 == 1:
            # x coordinate
            p = p * width
        else:
            # y coordinate
            p = p * hight
    print(points)
    x1, y1, x2, y2 = points

    exit()
    # Zeichne das Rechteck auf das Bild
    cv2.rectangle(image, (x1,y1), (x2,y2), color=color, thickness=thick)

while True: 
    draw_rectangle(img, results[0].boxes.xyxyn)
    img = results[0].orig_img
    cv2.imshow("Image", img)

    if cv2.waitKey(1) == 27:
        break

input()