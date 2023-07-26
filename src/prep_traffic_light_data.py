import custom_utils as util
from ultralytics import YOLO

############# Step 0: get images 
imgs = util.get_images("./dataset/raw/traffic_lights")
# print(len(imgs))

############# Step 1: analyze images with yolo

model = YOLO("yolov8n.pt")
model.predict(imgs, save=True, save_txt = True)

############# Step 2: extract all traffic lights from all analyzed images and create for each an new image
############# Step 3: check images manually and label them
#img = cv2.resize(img, (640,640))