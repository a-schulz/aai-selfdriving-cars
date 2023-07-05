#!/usr/bin/env python
# coding: utf-8

# In[22]:


import os

import cv2
import numpy as np
from dotenv import load_dotenv

# Classification for the base classes

load_dotenv()

# Specify the path to the dataset directory
dataset_dir = os.getenv('DATASET_DIR')

target_size = (128, 128)


# In[23]:


# Get all images
def read_images_from_directory(directory):
    images = []
    labels = []
    class_names = []
    path = []
    for root, dirs, files in os.walk(directory):
        for class_name in dirs:
            class_directory = os.path.join(root, class_name)
            for filename in os.listdir(class_directory):
                file_path = os.path.join(class_directory, filename)
                if os.path.isfile(file_path):
                    image = cv2.imread(file_path, cv2.COLOR_BGR2GRAY)
                    if image is not None:
                        image = cv2.resize(image, target_size)
                        images.append(image)
                        labels.append(os.path.basename(root) + "." + class_name)
                        path.append(file_path)
                        if class_name not in class_names:
                            class_names.append(class_name)

    images = np.array(images)
    labels = np.array(labels)
    class_names = np.array(class_names)
    path = np.array(path)

    return images, labels, class_names, path


# In[24]:


images, labels, class_names, path = read_images_from_directory(dataset_dir)
images_with_labels = list(zip(images, labels))
print("Number of images:", len(images))
print("Number of labels:", len(labels))
print("Class names:", class_names)
print("Labels:", labels)


# In[25]:


print("First image: ", images_with_labels[0][0])
print("First image - label: ", images_with_labels[0][1])


# In[26]:


import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

def annotate_image(image_path):
    # Load the pre-trained VGG16 model
    model = VGG16()

    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    img_preprocessed = preprocess_input(np.expand_dims(img_resized, axis=0))

    # Make predictions on the preprocessed image
    predictions = model.predict(img_preprocessed)
    #     print(predictions)
    #     input('p a k ...')
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Annotate the image with rectangles around the predicted objects
    annotated_img = img.copy()
    for _, label, prob in decoded_predictions:
        label = label.replace('_', ' ')
        print(label)
    # # Display and save the annotated image
    # cv2.imshow("Annotated Image", annotated_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite("annotated_image.jpg", annotated_img)

for i in range(0, len(path)):
    print(labels[i])
    annotate_image(path[i])


# In[27]:


# test one image
#img_path= "C:\Users\schul\OneDrive - HTW Dresden\Documents\6._Semester\Applied_AI\aai-selfdriving-cars\src\dataset\unsere-autorin-faehrt-viel-rad.jpg"
img_path = "./dataset/unsere-autorin-faehrt-viel-rad.jpg"
annotate_image(img_path)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
train_images, test_images, train_labels0, test_labels0 = train_test_split(images, labels, test_size=0.2,
                                                                          random_state=42)

# Convert the labels to one-hot encoded vectors
num_classes = len(np.unique(labels))

print(num_classes)

label_encoder = LabelEncoder()
train_labels1 = label_encoder.fit_transform(train_labels0)
test_labels1 = label_encoder.fit_transform(test_labels0)

from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(train_labels1, num_classes)
test_labels = to_categorical(test_labels1, num_classes)


# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

# Add convolutional and pooling layers
# Fixme
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(target_size[0], target_size[1], 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add more convolutional and pooling layers if desired

# Flatten the output from the previous layer
model.add(Flatten())

# Add fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


num_epochs = 1
batch_size = 10

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)


# In[ ]:


# Train the model
model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size,
          validation_data=(test_images, test_labels))


# In[ ]:


# Evaluate the model
loss, accuracy = model.evaluate(test_images, test_labels)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

