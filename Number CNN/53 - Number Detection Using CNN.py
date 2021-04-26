import os
import cv2
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

from sklearn.model_selection import train_test_split

################# PARAMETERS #####################
path = 'myDataNumber'
labelFile = 'label_numbers.csv'  # file with all names of classes

testRatio = 0.2  # if 1000 images split will 200 for testing
validationRatio = 0.2  # if 1000 images 20% of remaining 800 will be 160 for validation

imageDimesions = (32, 32, 3)

batchSizeVal = 50  # how many to process together
stepsPerEpochVal = 2000
epochsVal = 10
##################################################

################### IMPORT DATA ##################
images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
noOfClasses = len(myList)
print("Importing Classes.....")
for x in range(0, len(myList)):
    myPicList = os.listdir(path + "/" + str(x))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(x) + "/" + y)
        curImg = cv2.resize(curImg, (32, 32))
        images.append(curImg)
        classNo.append(x)
    print(x, end=" ")
print(" ")
print("Total Images in Images List = ", len(images))
print("Total IDS in classNo List= ", len(classNo))
##################################################

########### NUMPY ARRAY CONVERTION ###############
images = np.array(images)
classNo = np.array(classNo)
print(images.shape)
##################################################

################ DATA SPLIT ######################
x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validationRatio)
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)
##################################################

#### NUMBER OF IMAGES AND LABELS MATCH CHECK #####
print("Data Shapes")
print("Train", end="");
print(x_train.shape, y_train.shape)
print("Validation", end="");
print(x_validation.shape, y_validation.shape)
print("Test", end="");
print(x_test.shape, y_test.shape)
assert (x_train.shape[0] == y_train.shape[
    0]), "The number of images in not equal to the number of lables in training set"
assert (x_validation.shape[0] == y_validation.shape[
    0]), "The number of images in not equal to the number of lables in validation set"
assert (x_test.shape[0] == y_test.shape[0]), "The number of images in not equal to the number of lables in test set"
assert (x_train.shape[1:] == (imageDimesions)), " The dimesions of the Training images are wrong "
assert (x_validation.shape[1:] == (imageDimesions)), " The dimesionas of the Validation images are wrong "
assert (x_test.shape[1:] == (imageDimesions)), " The dimesionas of the Test images are wrong"
##################################################

######### READ CSV FILE ##########################
data = pd.read_csv(labelFile)
print("data shape ", data.shape, type(data))
##################################################

###### SAMPLE IMAGES DISPLAY OF ALL CLASSES ######
num_of_samples = []
cols = 5
num_classes = noOfClasses
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = x_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row["Name"])
            num_of_samples.append(len(x_selected))
##################################################

# ############## DISPLAY A BAR CHART ###############
# print(num_of_samples)
# plt.figure(figsize=(10, 5))
# plt.bar(range(0, num_classes), num_of_samples)
# plt.title("Distribution of the training dataset")
# plt.xlabel("Class number")
# plt.ylabel("Number of images")
# plt.show()
# ##################################################

########## IMAGE PREPROCESSING ###################
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)  # CONVERT TO GRAYSCALE
    img = equalize(img)  # STANDARDIZE THE LIGHTING IN AN IMAGE
    img = img / 255  # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
    return img

x_train = np.array(list(map(preprocessing, x_train)))  # TO IRETATE AND PREPROCESS ALL IMAGES
x_validation = np.array(list(map(preprocessing, x_validation)))
x_test = np.array(list(map(preprocessing, x_test)))
cv2.imshow("GrayScale Images",
           x_train[random.randint(0, len(x_train) - 1)])  # TO CHECK IF THE TRAINING IS DONE PROPERLY

# ADD A DEPTH OF 1
X_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
X_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 1)
X_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

############ AUGMENTATAION OF IMAGES #############
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             # 0.1 = 10%     IF MORE THAN 1 E.G 10 THEN IT REFFERS TO NO. OF  PIXELS EG 10 PIXELS
                             height_shift_range=0.1,
                             zoom_range=0.2,  # 0.2 MEANS CAN GO FROM 0.8 TO 1.2
                             shear_range=0.1,  # MAGNITUDE OF SHEAR ANGLE
                             rotation_range=10)  # DEGREES
dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train,
                       batch_size=20)  # REQUESTING DATA GENRATOR TO GENERATE IMAGES  BATCH SIZE = NO. OF IMAGES CREAED EACH TIME ITS CALLED
X_batch, y_batch = next(batches)

# TO SHOW AGMENTED IMAGE SAMPLES
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimesions[0], imageDimesions[1]))
    axs[i].axis('off')
plt.show()

y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
##################################################

######## CONVOLUTION NEURAL NETWORK MODEL ########
def myModel():
    no_Of_Filters = 60
    size_of_Filter = (5, 5)  # THIS IS THE KERNEL THAT MOVE AROUND THE IMAGE TO GET THE FEATURES.
    # THIS WOULD REMOVE 2 PIXELS FROM EACH BORDER WHEN USING 32 32 IMAGE
    size_of_Filter2 = (3, 3)
    size_of_pool = (2, 2)  # SCALE DOWN ALL FEATURE MAP TO GERNALIZE MORE, TO REDUCE OVERFITTING
    no_Of_Nodes = 500  # NO. OF NODES IN HIDDEN LAYERS
    model = Sequential()
    model.add((Conv2D(no_Of_Filters, size_of_Filter, input_shape=(imageDimesions[0], imageDimesions[1], 1),
                      activation='relu')))  # ADDING MORE CONVOLUTION LAYERS = LESS FEATURES BUT CAN CAUSE ACCURACY TO INCREASE
    model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))  # DOES NOT EFFECT THE DEPTH/NO OF FILTERS

    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(Dropout(0.5))  # INPUTS NODES TO DROP WITH EACH UPDATE 1 ALL 0 NONE
    model.add(Dense(noOfClasses, activation='softmax'))  # OUTPUT LAYER
    # COMPILE MODEL
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
##################################################

################### TRAIN ########################
model = myModel()
print(model.summary())
history = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=batchSizeVal),
                              steps_per_epoch=stepsPerEpochVal, epochs=epochsVal,
                              validation_data=(X_validation, y_validation), shuffle=1)
##################################################

################# PLOT ###########################
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])
##################################################

###### STORE THE MODEL AS A PICKLE OBJECT ########
pickle_out = open("model_trained_number.p", "wb")  # wb = WRITE BYTE
pickle.dump(model, pickle_out)
pickle_out.close()
cv2.waitKey(0)
##################################################