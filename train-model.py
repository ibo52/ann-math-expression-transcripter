import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l2
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

import os
import numpy as np
from datetime import datetime as dt
from matplotlib import pyplot as plt

def updateLR(epoch):
    initial_learning_rate = 0.001
    dropEvery = 10
    factor = 0.5
    lr = initial_learning_rate * (factor ** np.floor((1 + epoch) / dropEvery))
    return float(lr)

class ANNClassifierModel:
    def __init__(self,name='transCripterModel.h5',
                 data_path="",img_input_shape=(64,64,1),
                 batch_size=32, epoch=100):

        self.name=name if name.endswith(".h5") else name+".h5"

        self.img_rows, self.img_cols, self.channels = img_input_shape #model input shape

        self.colorMode="grayscale" if self.channels==1 else "rgb"#channel:RGB or grayscaled

        self.BATCH_SIZE = batch_size           # split data to subsets
        self.EPOCH=epoch                       #number of epochs to retrain

        self.train_path = ""

        if data_path=="":
            print("data path should be specified!!")&exit()
            
        for path in os.listdir(data_path):
            if "train" in path.lower():
                self.train_path=os.path.join(data_path,path)
                break
        
        self.labels = {}                       #labels of classified data
        self.elapsed_time= 0
        self.test_results=None

    def init_dataset(self):
            
        # artificially expanding our data size by augmenting method
        #and we splitting %18 as validation set
        train_datagen = ImageDataGenerator(
        validation_split=0.18,#data on train folder will split as train/validation by given ratio
        rescale=1. / 255,#normalize image array to increase performance
        #rotation_range=4,#do not over rotate, cause undesired behaviors for numbers or operators(x and + example)
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.1,#to be able to detect distorted images, cut small regiom of image form edges
        zoom_range=0.1,#same with above;to be able to detect distorted images
        fill_mode="nearest",
        #horizontal_flip=False# do not flip, may cause undesired behaviors on numbers
        )
        
        # images are passed through from augmenting method before supplying
        self.train_generator = train_datagen.flow_from_directory(
            self.train_path,
            color_mode=self.colorMode,  # since these are just numbers,no need color for anything
            target_size=(self.img_rows, self.img_cols),  # input image color
            batch_size=self.BATCH_SIZE,  # to efficiently run training on cpu
            class_mode='categorical',  # 10 numbers+6 operators=16 classes

            subset="training"
            # shuffle=True #no need to shufle
        )
        
        ##validation set takes %18 of train data
        self.validation_generator = train_datagen.flow_from_directory(
            self.train_path,
            color_mode=self.colorMode,  # since these are just numbers,no need color for anything
            target_size=(self.img_rows, self.img_cols),  # input image color
            batch_size=self.BATCH_SIZE,  # to efficiently run training on cpu
            class_mode='categorical',  # 10 numbers+6 operators=16 classes

            subset="validation"
            # shuffle=True #no need to shufle
        )

        #keep class labels on a dictionary
        self.labels=self.train_generator.class_indices

    def build_ANN(self):
        # -----BUILDING MODEL-----------------
        self.model = Sequential()
        # written network inspired some or more by lenet, since it provides good outputs

        INPUT_IM_SHAPE = self.img_rows, self.img_cols, self.channels

        F_MAP = 32  # number of feature maps for 1st and 2nd blocks
        F_MAP_3 = 64 # number of feature maps for 3rd block
        F_MAP_4 = 96
        KERNEL_SIZE = (3, 3)  # size of kernel to browse throuh image to determine filter weights
        POOL_SIZE = (2, 2)  # to reduce complexity,shrink feature outputs to half(2,2)

        NUM_CLASSES = len(self.labels)  #

        KERNEL_INITIALIZER = "glorot_uniform"
        REGULARIZER = l2(0.01)
        
        # Block-1
        self.model.add(Conv2D(F_MAP, KERNEL_SIZE, padding='same', kernel_initializer=KERNEL_INITIALIZER,
                         input_shape=INPUT_IM_SHAPE,
                         name="C1", activity_regularizer=REGULARIZER))
        self.model.add(Activation('relu', name="S1"))
        self.model.add(BatchNormalization())  # no need to normalize since data set is not so much
        self.model.add(MaxPooling2D(pool_size=POOL_SIZE, strides=POOL_SIZE))

        # block 2
        self.model.add(Conv2D(F_MAP, KERNEL_SIZE, padding='same', kernel_initializer=KERNEL_INITIALIZER,
                         name="C2", activity_regularizer=REGULARIZER))
        self.model.add(Activation('relu', name="S2"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=POOL_SIZE, strides=POOL_SIZE))
        # model.add(Dropout(0.2))

        # block 3
        self.model.add(Conv2D(F_MAP_3, KERNEL_SIZE, padding='same', kernel_initializer=KERNEL_INITIALIZER,
                         name="C3", activity_regularizer=REGULARIZER))
        self.model.add(Activation('relu', name="S3"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=POOL_SIZE, strides=POOL_SIZE))
        # model.add(Dropout(0.2))

        # block 4
        self.model.add(Conv2D(F_MAP_4, KERNEL_SIZE, padding='same', kernel_initializer=KERNEL_INITIALIZER,
                         name="C4", activity_regularizer=REGULARIZER))
        self.model.add(Activation('relu', name="S4"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=POOL_SIZE, strides=POOL_SIZE))
        # model.add(Dropout(0.2))
        
        # output by FULLY CONNECTED
        self.model.add(Flatten())  # flatten the outputs before fully connect
        self.model.add(Dropout(0.4))  # to overcome overfitting problem, throw half of outputs

        self.model.add(Dense(256, activation="relu", kernel_initializer=KERNEL_INITIALIZER,
                        name="F1"))
        self.model.add(Dense(128, activation="relu", kernel_initializer=KERNEL_INITIALIZER,
                        name="F2"))
        self.model.add(Dense(NUM_CLASSES, activation="softmax", kernel_initializer=KERNEL_INITIALIZER,
                        name="F3"))

        # -----BUILDING MODEL-----------------
    def begin_train(self):
        self.check_for_any_saved_model()

        # ---------------
        self.checkpoint = ModelCheckpoint(self.name,
                                          monitor='val_loss',
                                          mode='min',
                                          save_best_only=True,
                                          verbose=1)

        self.earlystop = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=7,
                                       verbose=1,
                                       restore_best_weights=True
                                       )

        # reduces learning rate
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.5,
                                      patience=3,
                                      verbose=1,
                                      min_delta=0.001)

        # callbacks = [earlystop,checkpoint,reduce_lr]
        self.callbacks = [self.checkpoint, self.earlystop, self.reduce_lr]#LearningRateScheduler(updateLR)]

        # summary of model structure
        self.model.summary()

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(learning_rate=0.001),
                           metrics=['accuracy'])

        #EPOCH is how many times the data re-processed to train. High values may cause overfit
        # but EarlyStopping function setted. So it will determine when to stop
        # keras module says 'model.fit_generator' function is deprecated. We have to use 'Model.fit'
        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.validation_generator,
            epochs=self.EPOCH,
            batch_size=self.BATCH_SIZE,
            callbacks=self.callbacks,
            verbose=1
        )

    # -----COMPILING MODEL-----------------
    def check_for_any_saved_model(self):
        """load to continue training if there is a model
        trained before, or waiting to complete to train"""
        saved_model = os.path.join(os.getcwd(), self.name)

        if os.path.exists(saved_model):

            from keras.models import load_model

            print("Saved model found. Loading..")
            print("BE Careful! If you set new model to train, you have to delete this model")
            print("or it may cause error since it is old model")
            self.model = load_model(saved_model)
        else:
            print("No save model found. Pass..")

    def fit_model(self):
        self.elapsed_time = dt.now()

        self.init_dataset()
        self.build_ANN()
        self.begin_train()

        self.elapsed_time=dt.now()-self.elapsed_time
        print("Training complete. Elapsed time:",self.elapsed_time)
        self.plot_results()

    def plot_results(self):
        # plot results:
        #accuracies
        fig, axs = plt.subplots(2)

        labels_ = ["test set", "validation set"]

        fig.suptitle("estimated results of model")
        axs[0].plot(self.history.history['accuracy'], marker="o")
        axs[0].plot(self.history.history['val_accuracy'], marker="o")
        axs[0].set_title("Accuracy")

        axs[0].set(xlabel='Epoch', ylabel='percentage')
        
        #
        #
        ##losses
        axs[1].plot(self.history.history['loss'], marker="o")
        axs[1].plot(self.history.history['val_loss'], marker="o")
        axs[1].set_title("Loss")

        axs[1].set(xlabel='Epoch', ylabel='percentage')

        plt.subplots_adjust(wspace=1, hspace=0.5)
        fig.text(0,0,f"Train Time:{self.elapsed_time}")
        fig.legend(labels_)

        try:
            os.mkdir(os.path.join(os.getcwd(),"plots"))
        except FileExistsError:
            pass
        plt.savefig(fname=f"plots/{self.name}-results.png",format="png",dpi=300)
        plt.tight_layout()
        plt.show()

if __name__=="__main__":
    print("""
    You can define new model by inheriting base class,
    you simply have to override 'build_ANN(self):' function
    """)
    _Model=ANNClassifierModel(data_path="dataset/")
    _Model.fit_model()
    with open("labels.py","w") as f:
        f.write(f"labels_={_Model.labels}")
