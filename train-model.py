import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os
from tensorflow.keras.regularizers import l2

img_rows,img_cols=32,32 # images are 400,400; resize to 32,32 to reduce complexity
BATCH_SIZE=32 #split data to subsets

train_path="dataset/train"
#test_path="dataset/test"
#train_path="/home/ibrahim/TEMPO/ann-math-expression-transcripter/dataset/zip files/Dataset: Handwritten Digits and Operators_michel_heusser/CompleteImages/All data (Compressed)"
labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "add","div","eq","mul","sub"]

print('data path:',train_path)
#artificially expanding our data size by augmenting method
train_datagen = ImageDataGenerator(
    validation_split=0.18,#data on train folder will split as train/validation by given ratio
    rescale=1./255,
    #rotation_range=4,#do not over rotate, cause undesired behaviors for numbers or operators(x and + example)
    shear_range=0.1,#to be able to detect distorted images, cut small regiom of image form edges
    zoom_range=0.1,#same with above;to be able to detect distorted images
    width_shift_range=0.05,
    height_shift_range=0.05,
    fill_mode="nearest",
    #horizontal_flip=False# do not flip, may cause undesired behaviors on numbers
    )

#images are passed through from augmenting method before supplying
train_generator = train_datagen.flow_from_directory(
 train_path,
 color_mode='grayscale',#since these are just numbers,no need color for anything
 target_size=(img_rows,img_cols),#input image color
 batch_size=BATCH_SIZE,#to efficiently run training on cpu
 class_mode='categorical',#9 numbers+5 operators=14 classes

 subset="training"
 #shuffle=True #no need to shufle
 )
print("train\n------------------")

#to check builded model performance
validation_generator=train_datagen.flow_from_directory(
    train_path,
    color_mode='grayscale',#since these are just numbers,no need color for anything
    target_size=(img_rows,img_cols),#input image color
    batch_size=BATCH_SIZE,#to efficiently run training on cpu
    class_mode='categorical',#9 numbers+5 operators=14 classes
    
    subset="validation"
    )
print("validation\n------------------")
"""
validation_datagen = ImageDataGenerator(rescale=1./255)

test_generator = validation_datagen.flow_from_directory(
 validation_path,
 color_mode='grayscale',
 target_size=(img_rows,img_cols),
 batch_size=batch_size,
 class_mode='categorical',
 shuffle=True)
"""


#-----BUILDING MODEL-----------------
model = Sequential()
#written network inspired some or more by lenet, since it provides good outputs

INPUT_IM_SHAPE= img_rows, img_cols, 1

F_MAP=32    #number of feature maps for 1st and 2nd blocks
F_MAP_3=64  #number of feature maps for 3rd block
F_MAP_4=96
KERNEL_SIZE=(3,3) #size of kernel to browse throuh image to determine filter weights
POOL_SIZE=(2,2)#to reduce complexity,shrink feature outputs to half(2,2)

NUM_CLASSES=16 #0,1..9,+,-,*,/(10num + 4symbols)+2 bracket ekledim

KERNEL_INITIALIZER="glorot_uniform"
REGULARIZER = l2(0.01)

#Block-1
model.add(Conv2D( F_MAP, KERNEL_SIZE ,padding='same',kernel_initializer=KERNEL_INITIALIZER,
                 input_shape=INPUT_IM_SHAPE,
                  name="C1", activity_regularizer=REGULARIZER))
model.add(Activation('relu',name="S1"))
model.add(BatchNormalization())#no need to normalize since data set is not so much
model.add(MaxPooling2D( pool_size=POOL_SIZE ,strides=POOL_SIZE ))

#block 2
model.add(Conv2D( F_MAP, KERNEL_SIZE, padding='same',kernel_initializer=KERNEL_INITIALIZER,
                  name="C2", activity_regularizer=REGULARIZER))
model.add(Activation('relu', name="S2"))
model.add(BatchNormalization())
model.add(MaxPooling2D( pool_size=POOL_SIZE ,strides=POOL_SIZE ))
#model.add(Dropout(0.2))

#block 3
model.add(Conv2D( F_MAP_3, KERNEL_SIZE, padding='same',kernel_initializer=KERNEL_INITIALIZER,
                  name="C3", activity_regularizer=REGULARIZER))
model.add(Activation('relu', name="S3"))
model.add(BatchNormalization())
model.add(MaxPooling2D( pool_size=POOL_SIZE ,strides=POOL_SIZE ))
#model.add(Dropout(0.2))

#block 4
model.add(Conv2D( F_MAP_4, KERNEL_SIZE, padding='same',kernel_initializer=KERNEL_INITIALIZER,
                  name="C4", activity_regularizer=REGULARIZER))
model.add(Activation('relu', name="S4"))
model.add(BatchNormalization())
model.add(MaxPooling2D( pool_size=POOL_SIZE ,strides=POOL_SIZE ))
#model.add(Dropout(0.2))


#output by FULLY CONNECTED
model.add(Flatten())#flatten the outputs before fully connect
model.add(Dropout(0.5))#to overcome overfitting problem, throw half of outputs

model.add(Dense(256,activation="relu",kernel_initializer=KERNEL_INITIALIZER,
                name="F1"))
model.add(Dense(128,activation="relu",kernel_initializer=KERNEL_INITIALIZER,
                name="F2"))
model.add(Dense(NUM_CLASSES,activation="softmax",kernel_initializer=KERNEL_INITIALIZER,
                name="F3"))#tanh çok hatalı accuracy veriy sakın kullanma

#model.add(Activation('softmax'))#
#-----BUILDING MODEL-----------------

'''
def math_symbol_and_digits_recognition(input_shape=(32, 32, 1)):
    regularizer = l2(0.01)
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', 
                     kernel_initializer=glorot_uniform(seed=0), 
                     name='conv1', activity_regularizer=regularizer))
    model.add(Activation(activation='relu', name='act1'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', 
                     kernel_initializer=glorot_uniform(seed=0), 
                     name='conv2', activity_regularizer=regularizer))
    model.add(Activation(activation='relu', name='act2'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', 
                     kernel_initializer=glorot_uniform(seed=0), 
                     name='conv3', activity_regularizer=regularizer))
    model.add(Activation(activation='relu', name='act3'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(120, activation='relu', kernel_initializer=glorot_uniform(seed=0), name='fc1'))
    model.add(Dense(84, activation='relu', kernel_initializer=glorot_uniform(seed=0), name='fc2'))
    model.add(Dense(14, activation='softmax', kernel_initializer=glorot_uniform(seed=0), name='fc3'))
    
    optimizer = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model=math_symbol_and_digits_recognition(input_shape=(32, 32, 1))
'''
def updateLR(epoch):
    initial_learning_rate = 0.001
    dropEvery = 10
    factor = 0.5
    lr = initial_learning_rate*(factor**np.floor((1 + epoch)/dropEvery))
    return float(lr)



#-----COMPILING MODEL-----------------
from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import numpy as np

import os
    
def check_for_any_saved_model():
    """load to continue training if there is a model
    trained before, or waiting to complete to train"""

    global model
    
    saved_model=os.path.join(os.getcwd(),"BIG_DATA_transCripterModel.h5")
    
    if os.path.exists( saved_model ):

        from keras.models import load_model 

        print("Saved model found. Loading..")

        model=load_model(saved_model)
    else:
        print("pass load_model")

check_for_any_saved_model()

#---------------
checkpoint = ModelCheckpoint('BIG_DATA_transCripterModel.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

#reduces learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.5,
                              patience=3,
                              verbose=1,
                              min_delta=0.001)

#callbacks = [earlystop,checkpoint,reduce_lr]
callbacks = [checkpoint,LearningRateScheduler(updateLR)]

#summary of model structure
model.summary()

model.compile(loss='categorical_crossentropy',
 optimizer = Adam(lr=0.001),
 metrics=['accuracy'])

from datetime import datetime as dt
t1=dt.now()
# how many times data re-processed to train.high values cause overflow
#but EarlyStopping function setted. So it will determine when to stop
EPOCH=100

LAST_EPOCH=22#son tamamlanan epoch yeri, analog olarak değiştiriyom
#model.fit_generator deprecated olmuş. Model.fit kullanın diyor
history=model.fit(
 train_generator,
 validation_data=validation_generator,
 epochs=EPOCH,
 batch_size=BATCH_SIZE,
 callbacks=callbacks,
 verbose=2,
 initial_epoch=LAST_EPOCH
 )#validation_data=validation_generator)

t2=dt.now()
print("elapsed time:",t2-t1)

def plot_results():
    global history
    #plot results
    ##accuracies
    plt.figure("estimated results of model")
    plt.subplot(2,1,1)
    plt.plot(history.history['accuracy'])

    plt.subplot(2,1,2)
    plt.plot(history.history['val_accuracy'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    #
    #
    ##losses
    plt.figure("estimated loss results of model")
    plt.subplot(2,1,1)
    plt.plot(history.history['loss'])

    plt.subplot(2,1,2)
    plt.plot(history.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
plot_results
