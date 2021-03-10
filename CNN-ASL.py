#!/usr/bin/env python
# coding: utf-8

# # 資料來源：Kaggle ASL Alphabet 
# # Image data set for alphabets in the American Sign Language
# ## https://www.kaggle.com/grassknoted/asl-alphabet?select=asl_alphabet_test

# In[11]:


#Import Library
import os
import cv2
import keras
import numpy as np
from time import time
from tensorflow.keras import utils

#From tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from numpy import mean
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Review Data
train_asl = 'asl_alphabet_train'
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
            'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
            'W', 'X', 'Y', 'Z', 'nothing', 'space', 'del']

plt.figure(figsize=(11, 11))
for i in range (0,29):
    plt.subplot(7,7,i+1)
    plt.xticks([])
    plt.yticks([])
    path = train_asl + "/{0}/{0}1.jpg".format(classes[i])
    img = plt.imread(path)
    plt.imshow(img)
    plt.xlabel(classes[i])

test_asl = 'asl_alphabet_test'
for i in range (0,29):
    plt.subplot(7,7,i+1)
    plt.xticks([])
    plt.yticks([])
    path = test_asl + "/{0}_test.jpg".format(classes[i])
    img2 = plt.imread(path)
    plt.imshow(img2)
    plt.xlabel(classes[i])

#Calculate Filennumbers
filenumber = []
for i in range (0,29):
    path = 'asl_alphabet_train/'+classes[i]
    filenum = 0
    for lists in os.listdir(path):
        sub_path = os.path.join(path, lists)
        if os.path.isfile(sub_path):
            filenum = filenum+1
    filenumber.append(filenum)
print(filenum)

get_ipython().run_line_magic('matplotlib', '')
plt.bar(classes,filenumber,
        color = 'c', 
        width = 0.6,
        tick_label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K','L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V','W', 'X', 'Y', 'Z', 'no', 'spa', 'del'])


plt.title('asl_alphabet_train')
plt.xlabel('Classes')
plt.ylabel('File Numbers')


# In[3]:


#loading Data
def load_data(train_asl):
    images = []
    labels = []
    images2 = []
    labels2 = []
    size = 32,32
    index = -1
    index2 = -1
    for folder in os.listdir(train_asl):
        index +=1
        for image in os.listdir(train_asl + "/" + folder):
            temp_img = cv2.imread(train_asl + '/' + folder + '/' + image)
            temp_img = cv2.resize(temp_img, size)
            images.append(temp_img)
            labels.append(index)
            
    for image in os.listdir(test_asl + "/"):
        index2 +=1
        temp_img2 = cv2.imread(test_asl + '/' + image)
        temp_img2 = cv2.resize(temp_img2, size)
        images2.append(temp_img2)
        labels2.append(index2)
    
    images = np.array(images)
    images = images.astype('float32')/255.0
    labels = utils.to_categorical(labels)
    images2 = np.array(images2)
    images2 = images2.astype('float32')/255.0
    labels2 = utils.to_categorical(labels2)
    
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.1)
    
    print('Loaded', len(x_train),'images for x training,','Train data shape =', x_train.shape)
    print('Loaded', len(x_test),'images for x testing','Test data shape =', x_test.shape)
    return x_train, x_test, y_train, y_test

start = time()
x_train, x_test, y_train, y_test = load_data(train_asl)
print('Loading:', time() - start)


# In[4]:


#Set Model 1 Layers
model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='sigmoid'))
model.add(Dense(len(classes), activation='softmax'))


batch = 64
epochs = 20
learning_rate = 0.001

#Training Model 1
adam = keras.optimizers.Adam(lr=learning_rate)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
start = time()
history = model.fit(x_train, y_train, batch_size=batch, epochs=epochs, validation_split=0.1, shuffle = True, verbose=1)
model.summary()
train_time = time() - start

#Iteration History
plt.figure(figsize=(12, 12))
plt.subplot(5, 5, 1)
plt.plot(history.history['accuracy'], label = 'train_accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.subplot(3, 2, 2)
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'validation')
plt.title('Model 1 (ConV activation)')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[5]:


#Set Model 2 Layers
model2 = Sequential()

model2.add(Conv2D(48, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(128, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(BatchNormalization())
model2.add(Flatten())
model2.add(Dropout(0.2))
model2.add(Dense(512, activation='sigmoid'))
model2.add(Dense(len(classes), activation='softmax'))

#Training Model 2
adam = keras.optimizers.Adam(lr=learning_rate)
model2.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
start = time()
history = model2.fit(x_train, y_train, batch_size=batch, epochs=epochs, validation_split=0.1, shuffle = True, verbose=1)
model2.summary()
train_time = time() - start

#Iteration History
plt.figure(figsize=(12, 12))
plt.subplot(5, 5, 1)
plt.plot(history.history['accuracy'], label = 'train_accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.subplot(3, 2, 2)
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'validation')
plt.title('model 2 (ConV activation)')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[6]:


#Set Model 3 Layers
model3 = Sequential()
model3.add(Conv2D(10, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Conv2D(16, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Conv2D(4, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(BatchNormalization())
model3.add(Flatten())
model3.add(Dropout(0.2))
model3.add(Dense(128, activation='sigmoid'))
model3.add(Dense(len(classes), activation='softmax'))

#Training Model 3
adam = keras.optimizers.Adam(lr=learning_rate)
model3.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
start = time()
history3 = model3.fit(x_train, y_train, batch_size=batch, epochs=epochs, validation_split=0.1, shuffle = True, verbose=1)
model3.summary()
train_time = time() - start

#Iteration History
plt.figure(figsize=(12, 12))
plt.subplot(5, 5, 1)
plt.plot(history.history['accuracy'], label = 'train_accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.subplot(3, 2, 2)
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'validation')
plt.title('model 3 (ConV activation)')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[7]:


#Set Model 4 Layers
model4 = Sequential()
model4.add(Conv2D(48, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model4.add(MaxPooling2D(pool_size=(2, 2)))
model4.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model4.add(MaxPooling2D(pool_size=(2, 2)))
model4.add(Conv2D(128, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model4.add(MaxPooling2D(pool_size=(2, 2)))
model4.add(BatchNormalization())
model4.add(Flatten())
model4.add(Dropout(0.2))
model4.add(Dense(512, activation='softmax'))
model4.add(Dense(len(classes), activation='sigmoid'))

#Training Model 4
adam = keras.optimizers.Adam(lr=learning_rate)
model4.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
start = time()
history = model4.fit(x_train, y_train, batch_size=batch, epochs=epochs, validation_split=0.1, shuffle = True, verbose=1)
model4.summary()
train_time = time() - start

#Iteration History
plt.figure(figsize=(12, 12))
plt.subplot(5, 5, 1)
plt.plot(history.history['accuracy'], label = 'train_accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.subplot(3, 2, 2)
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'validation')
plt.title('model 4 (ConV activation)')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[8]:


#Set Model 5 Layers
model5 = Sequential()
model5.add(Conv2D(48, (3, 3), padding='same', input_shape=(32, 32, 3), activation='sigmoid'))
model5.add(MaxPooling2D(pool_size=(2, 2)))
model5.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3), activation='sigmoid'))
model5.add(MaxPooling2D(pool_size=(2, 2)))
model5.add(Conv2D(128, (3, 3), padding='same', input_shape=(32, 32, 3), activation='sigmoid'))
model5.add(MaxPooling2D(pool_size=(2, 2)))
model5.add(BatchNormalization())
model5.add(Flatten())
model5.add(Dropout(0.2))
model5.add(Dense(512, activation='sigmoid'))
model5.add(Dense(len(classes), activation='softmax'))

#Training Model 5
adam = keras.optimizers.Adam(lr=learning_rate)
model5.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
start = time()
history = model5.fit(x_train, y_train, batch_size=batch, epochs=epochs, validation_split=0.1, shuffle = True, verbose=1)
model5.summary()
train_time = time() - start

#Iteration History
plt.figure(figsize=(12, 12))
plt.subplot(5, 5, 1)
plt.plot(history.history['accuracy'], label = 'train_accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.subplot(3, 2, 2)
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'validation')
plt.title('model 5 (ConV activation)')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[9]:


#Set Model 6 Layers
model6 = Sequential()
model6.add(Conv2D(48, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model6.add(MaxPooling2D(pool_size=(2, 2)))
model6.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model6.add(MaxPooling2D(pool_size=(2, 2)))
model6.add(Conv2D(128, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model6.add(MaxPooling2D(pool_size=(2, 2)))
model6.add(BatchNormalization())
model6.add(Flatten())
model6.add(Dropout(0.2))
model6.add(Dense(512, activation='sigmoid'))
model6.add(Dense(len(classes), activation='softmax'))

#Training Model 6
adagrad = keras.optimizers.Adam(lr=learning_rate)
model6.compile(optimizer=adagrad, loss='categorical_crossentropy', metrics=['accuracy'])
start = time()
history = model6.fit(x_train, y_train, batch_size=batch, epochs=epochs, validation_split=0.1, shuffle = True, verbose=1)
model6.summary()
train_time = time() - start

#Iteration History
plt.figure(figsize=(12, 12))
plt.subplot(5, 5, 1)
plt.plot(history.history['accuracy'], label = 'train_accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.subplot(3, 2, 2)
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'validation')
plt.title('model 6 (ConV activation)')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[10]:


#Evaluate Model with Test Data
scores = model.evaluate(x_test, y_test)
print('Model 1 CNN Accuracy : %.3f ' % (scores[1]))
scores = model2.evaluate(x_test, y_test)
print('Model 2 CNN Accuracy : %.3f ' % (scores[1]))
scores = model3.evaluate(x_test, y_test)
print('Model 3 CNN Accuracy : %.3f ' % (scores[1]))
scores = model4.evaluate(x_test, y_test)
print('Model 4 CNN Accuracy : %.3f ' % (scores[1]))
scores = model5.evaluate(x_test, y_test)
print('Model 5 CNN Accuracy : %.3f ' % (scores[1]))
scores = model6.evaluate(x_test, y_test)
print('Model 6 CNN Accuracy : %.3f ' % (scores[1]))

