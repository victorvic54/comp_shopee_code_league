# We use tensorflow==2.2.0
# We use keras==2.3.1

import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from keras.utils import np_utils

import scipy.io
import scipy.misc

import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint
import efficientnet.keras as efn
from imblearn.over_sampling import SMOTE

distribution = {}

SIZE=350
batch_size = 8
maxepoches = 5
learning_rate = 0.001

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

import keras.optimizers as Optimizer
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


base_model_1 = efn.EfficientNetB7(include_top=False,weights='imagenet',classes=42,input_shape=(SIZE,SIZE,3))

model_1 = keras.Sequential()
model_1.add(base_model_1)
model_1.add(keras.layers.GlobalAveragePooling2D())
model_1.add(keras.layers.Dropout(0.25))
model_1.add(keras.layers.Dense(42, activation=('softmax')))

sgd = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
model_1.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


directory = "shopee-product-detection-dataset/train/train/"

for inc in range(40):
    print("=========================")
    print("Iteration:" + str(inc + 1))
    
    Images = []
    Labels = []
    
    for labels in os.listdir(directory):
        total = len(os.listdir(directory + labels))
        
        start = inc * (total // 40)
        last = (inc + 1) * (total // 40)
        
        for image_file in sorted(os.listdir(directory + labels))[start:last]:
            
            if (image_file[-4:] == ".tmp"):
                print("Skipping .tmp file")
                continue
            
            image = cv2.imread(directory + labels + '/' + image_file)
            image_size = image.shape
            
            if (image_size == (1, 1, 3)):
                print(directory + labels + '/' + image_file + " is too small")
                continue
            
            image = cv2.resize(image,(SIZE,SIZE)) # Resize the image, Some images are different sizes. (Resizing is very Important)
            Images.append(image)
            Labels.append(int(labels))
        
            Images, Labels = shuffle(Images, Labels, random_state=817328462) # Shuffle the dataset you just prepared.


    Images = np.array(Images) / 255 # converting the list of images to numpy array.
    Labels = np.array(Labels)
    
    num_shape_layer = Images.shape[0]

    print("Shape of Images:",Images.shape)
    print("Shape of Labels:",Labels.shape)

    Images = Images.reshape(Images.shape[0], SIZE * SIZE * 3)

    print("=============================")
    print("Shape of Reshaped Images:",Images.shape)

    x_train, x_test, y_train, y_test = train_test_split(Images, Labels, test_size = 0.08)

    print("Num train_imgs", x_train.shape)
    print("Num y_train", y_train.shape)

    smt = SMOTE(random_state=42)
    x_train_smote, y_train_smote = smt.fit_resample(x_train, y_train)

    print("=============================")
    print("Shape of new Images:", x_train_smote.shape)
    print("Shape of new Labels:", y_train_smote.shape)
    
    x_train = x_train_smote.reshape(x_train_smote.shape[0], SIZE, SIZE, 3)
    y_train = y_train_smote

    x_test = x_test.reshape(x_test.shape[0], SIZE, SIZE, 3)

    print("Newest train shape:", x_train.shape)
    print("Newest train label:", y_train.shape)
    print("=============================")

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    #data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=35,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.25,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.25,  # randomly shift images vertically (fraction of total height)
        shear_range=0.05,
        zoom_range=0.15,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(x_train)

    mcp_save = ModelCheckpoint('model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5', save_best_only=True, monitor='val_loss', mode='min')
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-4, mode='min')

    # training process in a for loop with learning rate drop every 25 epoches.
    model_1.fit(datagen.flow(x_train, y_train,
                batch_size=batch_size),
                epochs=maxepoches,
                steps_per_epoch=x_train.shape[0] // batch_size,
                validation_data=(x_test, y_test),
                callbacks=[mcp_save, reduce_lr_loss, es],
                verbose=1)

    model_1.save('result_legend/data_final' + str(inc + 1) + '.h5')
    model_1.save_weights('result_legend/data_weight' + str(inc + 1) + '.h5')

print("==============================")
print("Stage one of training has done")
print("Starting stage two where we remember what we learn with 1 epoch")
print("==============================")


K.set_value(model_1.optimizer.lr, 0.0008)
maxepoches = 1

directory = "shopee-product-detection-dataset/train/train/"

for inc in range(40):
    print("=========================")
    print("Iteration:" + str(inc + 1))
    
    Images = []
    Labels = []
    
    for labels in os.listdir(directory):
        total = len(os.listdir(directory + labels))
        
        start = inc * (total // 40)
        last = (inc + 1) * (total // 40)
        
        for image_file in sorted(os.listdir(directory + labels))[start:last]:
            
            if (image_file[-4:] == ".tmp"):
                print("Skipping .tmp file")
                continue
            
            image = cv2.imread(directory + labels + '/' + image_file)
            image_size = image.shape
            
            if (image_size == (1, 1, 3)):
                print(directory + labels + '/' + image_file + " is too small")
                continue
            
            image = cv2.resize(image,(SIZE,SIZE)) # Resize the image, Some images are different sizes. (Resizing is very Important)
            Images.append(image)
            Labels.append(int(labels))
        
            Images, Labels = shuffle(Images, Labels, random_state=817328462) # Shuffle the dataset you just prepared.


    Images = np.array(Images) / 255 # converting the list of images to numpy array.
    Labels = np.array(Labels)
    
    num_shape_layer = Images.shape[0]

    print("Shape of Images:",Images.shape)
    print("Shape of Labels:",Labels.shape)

    Images = Images.reshape(Images.shape[0], SIZE * SIZE * 3)

    print("=============================")
    print("Shape of reshaped Images:", Images.shape)

    x_train, x_test, y_train, y_test = train_test_split(Images, Labels, test_size = 0.08)

    print("Num train_imgs", x_train.shape)
    print("Num y_train", y_train.shape)

    smt = SMOTE(random_state=42)
    x_train_smote, y_train_smote = smt.fit_resample(x_train, y_train)

    print("=============================")
    print("Shape of new Images:", x_train_smote.shape)
    print("Shape of new Labels:", y_train_smote.shape)

    x_train = x_train_smote.reshape(x_train_smote.shape[0], SIZE, SIZE, 3)
    y_train = y_train_smote

    x_test = x_test.reshape(x_test.shape[0], SIZE, SIZE, 3)

    print("Newest train shape:", x_train.shape)
    print("Newest train label:", y_train.shape)
    print("=============================")
    
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    #data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=35,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.25,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.25,  # randomly shift images vertically (fraction of total height)
        shear_range=0.05,
        zoom_range=0.15,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(x_train)

    mcp_save = ModelCheckpoint('model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5', save_best_only=True, monitor='val_loss', mode='min')
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-4, mode='min')

    # training process in a for loop with learning rate drop every 25 epoches.
    model_1.fit(datagen.flow(x_train, y_train,
                batch_size=batch_size),
                epochs=maxepoches,
                steps_per_epoch=x_train.shape[0] // batch_size,
                validation_data=(x_test, y_test),
                callbacks=[mcp_save, reduce_lr_loss, es],
                verbose=1)

    model_1.save('warrior_result/data_final' + str(inc + 1) + '.h5')
    model_1.save_weights('warrior_result/data_weight' + str(inc + 1) + '.h5')


# Modify this for the test file that you want to predict
test = pd.read_csv("../input/cookies/test.csv")
test_img_dirs = test['filename'].tolist()

indexes = [(0,250),(250,500),(500,750), (750,1000),
           (1000,1250),(1250,1500),(1500,1750),(1750,2000),
           (2000,2250),(2250,2500),(2500,2750),(2750,3000),
           (3000,3250),(3250,3500),(3500,3750),(3750,4000),
           (4000,4250),(4250,4500),(4500,4750),(4750,5000),
           (5000,5250),(5250,5500),(5500,5750),(5750,6000),
           (6000,6250),(6250,6500),(6500,6750),(6750,7000),
           (7000,7250),(7250,7500),(7500,7750),(7750,8000),
           (8000,8250),(8250,8500),(8500,8750),(8750,9000),
           (9000,9250),(9250,9500),(9500,9750),(9750,10000),
           (10000,10250),(10250,10500),(10500,10750),(10750,11000),
           (11000,11250),(11250,11500),(11500,11750),(11750,12000),
           (12000,12186)]

yhats = []

# This is the location where you save your previous data
model = load_model("warrior_result/data_final40.h5")
print("Model loaded")

pred = []
for ind in indexes:
    test_cat = np.zeros((250, 1))
    test_img = np.zeros((250, 350, 350, 3))
    i = 0
    for imgdir in test_img_dirs[ind[0] : ind[1]]:
        test_img[i,...] = cv2.resize(cv2.imread("../input/cookies/test/test/" + imgdir), dsize=(350, 350))
        i += 1
    print(i, end = " ")
    test_img = test_img[:i]
    test_img /= 255

    pred += [np.argmax(pred) for pred in model.predict(test_img)]
    print(len(pred))

test.drop(columns=['category'])
test['category'] = pd.Series(pred).apply(lambda x: ("0" + str(x))[-2:])
test.to_csv("mythic_weapon.csv", index = False)


'''

By right we improve our accuracy using ensembling with other models.
We took the mode of the combination of several models with weights.
Example:

6 x 0.80... EfficientNet 350 
4 x 0.78965 EfficientNet 350
2 x 0.77355 EfficientNet 299
2 x 0.75851 EfficientNet 299
2 x 0.74689 EfficientNet 256
1 x 0.73291 Xception
1 x 0.72340 Xception

I will make an improvement to 81.3
Here is the snippet of the code (after loading all the csv results) named
`hello1`, `hello2`, `hello3`, ... and the test csv file called `test`


new_pred = []
count = 0

for i in range(12186):
    lst = []
    a = [hello1["category"][i]] * 1
    b = [hello2["category"][i]] * 1
    c = [hello3["category"][i]] * 1
    d = [hello4["category"][i]] * 2
    e = [hello5["category"][i]] * 2
    f = [hello6["category"][i]] * 2
    g = [hello7["category"][i]] * 4
    h = [hello8["category"][i]] * 6
    lst = lst + a + b + c + d + e + f + g +h
    
    expected = hello8["category"][i]
    new_val = max(set(lst), key=lst.count)
    
    if (new_val != expected):
        count += 1
        print(i)
    
    new_pred.append(new_val)
    
test.drop(columns=['category'])
test['category'] = pd.Series(new_pred).apply(lambda x: ("0" + str(x))[-2:])
test.to_csv("new_legendary_weapon.csv", index = False)

'''

'''

What we have done great:
- We filter some confusing input data (not so many though)
- We data augment our images
- We use notable base model such as efficientnet and xception
- We resize our image into quite a large value: 350 x 350
- We use SMOTE (reduce bias)
- We use fit_generator callbacks such as ReduceLROnPlateau (min_lr is so important since we divide our data into pieces instead of all)

Things for improvement:
- Surely the optimal value of learning rate will greatly increase your accuracy (need to try next time)
- The capability to freeze some of the base model for some portion (need to test too)
- Probably adding some layers after loading of base models (Example: dense, batch normalization, and dropout)
- Should add more different base model so that the process of ensembling will be more better
- Maybe adding more epoch (12 maybe) but adjust the learning rate to be lower a bit (0.0005). For every iteration, you will need to reset the learning rate 
  into 0.0005 again (since we divided our data into pieces) because ReduceLROnPlateau will decrease your learning rate as your epoch goes on
- They said bigger image size will be better (memory will be compromised) since the data size are around 700 - 800, there is even a 1000+
- Probably hand filtering the train and test data will help a lot

'''