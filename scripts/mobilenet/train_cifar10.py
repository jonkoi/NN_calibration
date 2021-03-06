# Training procedure for LeNet-5 CIFAR-10.
#Code base from https://github.com/BIGBALLON/cifar-10-cnn/blob/master/1_Lecun_Network/LeNet_dp_da_wd_keras.py

import keras
import numpy as np
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from sklearn.model_selection import train_test_split
import pickle
import string
import random

rep = 5

batch_size    = 128
epochs        = 300
iterations    = 45000 // batch_size
num_classes   = 10
weight_decay  = 0.0001
seed = 333
N = 1
print("N:", N)


log_filepath  = './lenet_dp_da_wd'

def id_generator(size=5, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def scheduler(epoch):
    if epoch <= 60:
        return 0.1
    if epoch <= 120:
        return 0.01
    if epoch <= 160:
        return 0.001
    return 0.0001

def color_preprocessing(x_train, x_val, x_test):

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')

    mean = np.mean(x_train, axis=(0,1,2))  # Per channel mean
    std = np.std(x_train, axis=(0,1,2))
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    return x_train, x_val, x_test

if __name__ == '__main__':

    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train45, x_val, y_train45, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)  # random_state = seed
    x_train45, x_val, x_test = color_preprocessing(x_train45, x_val, x_test)

    y_train45 = keras.utils.to_categorical(y_train45, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    id = id_generator()

    for i in range(rep):
        # build network
        model = keras.applications.mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=10)
        print(model.summary())

        # set callback
        change_lr = LearningRateScheduler(scheduler)
        cbks = [change_lr]

        # using real-time data augmentation
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(horizontal_flip=True,
                width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)


        datagen.fit(x_train45)

        # start traing
        hist = model.fit_generator(datagen.flow(x_train45, y_train45,batch_size=batch_size, shuffle=True),
                            steps_per_epoch=iterations,
                            epochs=epochs,
                            callbacks=cbks,
                            validation_data=(x_val, y_val))
        # save model
        model.save(id + '_' + str(i) + '_' + 'lenet_c10.h5')

        print("Get test accuracy:")
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("Test: accuracy1 = %f  ;  loss1 = %f" % (accuracy, loss))

        print("Pickle models history")
        with open(id + '_' + str(i) + '_' + 'hist_lenet_c10.p', 'wb') as f:
            pickle.dump(hist.history, f)

        K.clear_session()
