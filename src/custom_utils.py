import os
import cv2

def is_image(filename):
    # check if file is an image
    image_extensions = ['.jpg', '.jpeg', '.png']  
    _, file_extension = os.path.splitext(filename)
    return file_extension.lower() in image_extensions

def get_images(dir, count = -1):
    # load images which have to be evaluated from the ai into a list
    # images are in original size and color
    # if bound is positive: only first count files (not images) get analyzed
    # return value: elements with same index in the first and second array are the same image
    images = []
    paths = []
    for i, file in enumerate(os.listdir(dir)):
        if i == count:
            break
        path = os.path.join(dir, file)
        if os.path.isfile(path) and is_image(file):
            paths.append(path)

            # images contains images in RGB format
            img = cv2.imread(path)
            images.append(img)
    return paths, images

def process_txt_for_label(directory_path, label):
    data_dict = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
                data_list = []
                for line in lines:
                    values = line.split()
                    integer_value = int(values[0])
                    if integer_value == label:
                        data_list.append([float(value) for value in values[1:]])
                if data_list:
                    data_dict[filename[:-4]] = [label] + data_list
    return data_dict

def extract_rectangle_from_image(image, points, save=True, output_path="./"):
    """
    Function to extract a rectangle from an image.
    @param image:
    @param points:
    @param save:
    @param output_path:
    @return: Image as numpy array.
    """
    x1, y1, x2, y2 = points

    height, width = image.shape[:2]
    x1_pixel = int(x1 * width)
    y1_pixel = int(y1 * height)
    x2_pixel = int(x2 * width)
    y2_pixel = int(y2 * height)

    extracted_rect = image[y1_pixel:y2_pixel, x1_pixel:x2_pixel]
    
    if save:
        cv2.imwrite(output_path, extracted_rect)
    return extracted_rect

def write_images(images, path, main_name = ""):
    for idx, i in enumerate(images):
        cv2.imwrite(

def convert_dataset(categories, src_dir, dest_dir):
    # convert yolo v8 dataset format to which we used in lectures for image classification
    # the  order of categories have to be the same as in the yaml file
    for data_use in ["train", "valid", "test"]:
        
        # create folder for each category in dest_dir
        for c in categories:
            path = dest_dir + "/" + c
            if not os.path.exists(path):
                os.mkdir(path)

        # path where the txt files get stored to not destroy the box position
        dest_path_txt = dest_dir + "/__labels__"
        if not os.path.exists(dest_path_txt ):
            os.mkdir(dest_path_txt)

        txt_path = os.path.join(src_dir, data_use, "labels")
        for filename_txt in os.listdir(txt_path):
            if filename_txt.endswith(".txt"):
                txt_filepath = os.path.join(txt_path, filename_txt)
                with open(txt_filepath, 'r') as file:
                    lines = file.readlines()
                    label = int(lines[0].split()[0])  # Label aus der ersten Zeile extrahieren
                    dest_folder = ""
                    for i in range(0,len(categories)):
                        if label == i:
                            dest_folder = categories[i]

                    # Zielpfad für die Verschiebung erstellen

                    # Verschiebung der Datei in den entsprechenden Ordner
                    #os.rename(txt_filepath, dest_path_txt)
                    
                    #dest_path_tmp = os.path.join(dest_dir, dest_folder, filename_txt)
                    tmp = txt_filepath.replace(".txt",".jpg")
                    jpg_filepath = tmp.replace("labels", "images")

                    dest_path_jpg = os.path.join(dest_dir, dest_folder, filename_txt.replace(".txt",".jpg"))
                    os.rename(jpg_filepath, dest_path_jpg)

    for filename in os.listdir(src_dir):
        p = os.path.join(src_dir, filename)
        if filename.endswith(".yaml"):
            os.rename(p, dest_path_txt + "/" + filename)        
    # TODO clean up directory

if __name__ == "__main__":
#    convert_dataset(["green", "red", "yellow"],
#                    "/home/jay/module/ai_app/self_driving_cars/traffic_lights",
#                    "/home/jay/module/ai_app/self_driving_cars/aai-selfdriving-cars/dataset/traffic_light" )

    p, i =  get_images('/home/jay/module/ai_app/self_driving_cars/aai-selfdriving-cars/dataset/')

    print(p)
    print(i)