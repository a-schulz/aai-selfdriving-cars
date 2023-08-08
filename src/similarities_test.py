from skimage.metrics import structural_similarity as ssim
import custom_utils as cu
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from dotenv import load_dotenv
load_dotenv()
custom_data_dir = os.getenv('TRAFFIC_LIGHT_CUSTOM_DATA')
drop_data_dir = os.getenv('TRAFFIC_LIGHT_DROP_DATA')

labels = ["green", "red", "yellow"]

def find_similar(threshold = 0.6, directory = None, images = None):
    if directory is not None:
        # load pictures 
        _ , images = cu.get_images(directory)

    # Erzeuge eine leere Liste zum Speichern der Tupel
    similar_images = []
    num_images = len(images)

    # compare each picture
    for i in range(num_images):
        for j in range(i + 1, num_images):
            # convert images into grayscale (ssim works only with that)
            gray_image1 = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)#cv2.COLOR_RGB2GRAY)
            gray_image2 = cv2.cvtColor(images[j], cv2.COLOR_BGR2GRAY)

            # calculate ssim value
            similarity_score, _ = ssim(gray_image1, gray_image2, full=True)

            # check if ssim is bigger than threshold to find most similar pictures
            if similarity_score > threshold:
                similar_images.append((i, j, similarity_score))

    return np.array(similar_images)

def drop_similar(custom_dict = None, threshold = 0.6):

    sim_dic = {}
    for l in custom_dict:
        similar_images = []

        images = custom_dict[l][0]
        paths  =  custom_dict[l][1]
        
        num_images = len(images)

        # compare each picture
        for i in range(num_images):
            for j in range(i + 1, num_images):
                # convert images into grayscale (ssim works only with that)
                gray_image1 = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)#cv2.COLOR_RGB2GRAY)
                gray_image2 = cv2.cvtColor(images[j], cv2.COLOR_BGR2GRAY)

                # calculate ssim value
                similarity_score, _ = ssim(gray_image1, gray_image2, full=True)

                # check if ssim is bigger than threshold to find most similar pictures
                if similarity_score > threshold:
                    similar_images.append((i, j, similarity_score))
                    
                    # drop images[i] to other folder:
                    print(paths[i])
                    pa = os.path.join(drop_data_dir, l, os.path.basename(paths[i]))
                    print(pa)
                    #os.rename(paths[i], pa)
                    paths.pop(i)    # TODO damit wird die Länge verkürzt, daher wird irgendwann i index error haben
                    images.pop(i)

        custom_dict[l][0] = images 
        custom_dict[l][1] = paths  

        #sim_dic.update({l : })

        return np.array(similar_images)


def show_image_pairs(image_list, index_pairs):
    current_pair_index = 0
    num_pairs = len(index_pairs)

    while current_pair_index < num_pairs:
        image_indices = index_pairs[current_pair_index]
        image1 = image_list[image_indices[0]]
        image2 = image_list[image_indices[1]]

        combined_image = cv2.hconcat([image1, image2])

        cv2.imshow("Image Pair" + str(index_pairs[current_pair_index]) , combined_image)
        key = cv2.waitKey(0) 

        if key == ord("1"):
            current_pair_index += 1
        elif key == ord("2"):
            break
    cv2.destroyAllWindows()

cus_dic = {}
sim_dic = {}
for l in labels:
    custom_cls_path = os.path.join(custom_data_dir, l)
    paths , custom_images = cu.get_images(custom_cls_path)
    cus_dic.update({l : [custom_images, paths]})
    #s = find_similar(threshold=0.68, images=cus_dic[l][0])

    drop_similar(custom_dict=cus_dic, threshold=0.8)
    #sim_dic.update({l : s})

# threshold= 0.8
#sim_dic = {'green': np.array([[67, 75]]), 'red': np.array([[ 28,  32], [ 49,  53], [102, 103], [125, 154], [128, 155], [140, 145], [140, 154]]), 'yellow': np.array([[20, 21]])}

# threshold= 0.6
#sim_dic = {'green': np.array([[  3,  27], [  3, 114], [  5,  11], [  5,  15], [  5,  25], [  5,  99], [  5, 116], [  6,  17], [  6, 124], [ 11,  99], [ 11, 116], [ 15,  99], [ 16, 123], [ 17, 124], [ 25,  29], [ 25, 116], [ 27, 114], [ 32,  33], [ 34,  99], [ 34, 125], [ 67,  75], [ 77,  78], [ 77,  79], [ 78,  79], [ 99, 101], [ 99, 116], [106, 116], [116, 125], [122, 136]]),
#       'red': np.array([[  5, 142], [  6, 141], [ 15,  50], [ 17,  18], [ 17,  25], [ 17,  33], [ 17, 130], [ 17, 145], [ 18,  25], [ 18,  33], [ 18, 121], [ 18, 130], [ 18, 145], [ 18, 154], [ 22, 134], [ 22, 135], [ 25,  33], [ 25,  37], [ 25, 125], [ 25, 130], [ 25, 140], [ 25, 145], [ 25, 154], [ 27,  30], [ 28,  32], [ 28, 138], [ 28, 144], [ 30, 138], [ 30, 139], [ 32, 138], [ 32, 144], [ 33, 125], [ 33, 130], [ 33, 140], [ 33, 145], [ 33, 154], [ 34, 144], [ 49,  53], [ 49, 152], [ 49, 156], [ 51,  62], [ 51, 121], [ 51, 125], [ 51, 154], [ 52, 128], [ 52, 155], [ 53, 152], [ 53, 156], [ 62, 121], [ 62, 125], [ 62, 154], [ 68, 147], [ 72, 138], [ 72, 142], [ 72, 144], [ 89,  90], [ 91,  93], [ 98, 102], [102, 103], [102, 108], [103, 108], [120, 124], [122, 149], [125, 128], [125, 140], [125, 145], [125, 154], [128, 130], [128, 140], [128, 154], [128, 155], [129, 139], [129, 156], [130, 140], [130, 145], [130, 155], [134, 135], [138, 139], [138, 142], [138, 144], [138, 152], [139, 144], [139, 152], [139, 156], [140, 145], [140, 154], [141, 143], [142, 144], [145, 154], [147, 150], [152, 156], [154, 155]]), 
#       'yellow': np.array([[18, 19], [18, 20], [18, 21], [19, 22], [20, 21], [20, 22], [21, 22], [51, 53], [51, 73], [53, 73], [68, 69]])}


#print(sim_dic)


#for cls in labels:
#    show_image_pairs(cus_dic[cls][0], sim_dic[cls])