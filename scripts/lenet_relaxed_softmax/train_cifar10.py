# Training procedure for LeNet-5 CIFAR-10.
#Code base from https://github.com/BIGBALLON/cifar-10-cnn/blob/master/1_Lecun_Network/LeNet_dp_da_wd_keras.py
import sys
sys.path.append("../utility/")

import keras
import numpy as np
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Input, Lambda, Activation, Multiply, Add
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import pickle
import string
import random
import os
from datalog import logInit
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# from relaxed_softmax import RelaxedSoftmax

rep = 1

batch_size    = 128
epochs        = 300
iterations    = 45000 // batch_size
num_classes   = 10
weight_decay  = 0.0001
seed = 333
N = 1
print("N:", N)


log_filepath  = '/home/khoi/NN_calibration_results/lenet_relaxed_softmax_' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6)) + '/'
os.mkdir(log_filepath)

def id_generator(size=5, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def custom_loss(y_true, y_pred):
    # y_true = K.print_tensor(y_true, message='y_true = ')
    # y_pred = K.print_tensor(y_pred, message='y_pred = ')
    return K.categorical_crossentropy(y_true, y_pred)

def build_model(n=1, num_classes = 10, addition = False):
    """
    parameters:
        n: (int) scaling for model (n times filters in Conv2D and nodes in Dense)
    """
    inputs = Input(shape=(32,32,3))
    x = Conv2D(n*6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(inputs)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = BatchNormalization(epsilon=1.1e-5)(x)
    x = Conv2D(n*16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(epsilon=1.1e-5)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(n*120, activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay) )(x)
    x = Dense(n*84, activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay) )(x)
    if addition:
        x = Dense(num_classes + 2, activation = None, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay) )(x)
        a = Lambda(lambda x : x[:,0])(x)
        b = Lambda(lambda x : x[:,1])(x)
        logits = Lambda(lambda x : x[:,2:])(x)
        soft_logits = Multiply()([logits, a])
        soft_logits = Add()([soft_logits, b])
    else:
        x = Dense(num_classes + 1, activation = None, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay) )(x)
        temperature = Lambda(lambda x : x[:,0])(x)
        temperature = Lambda(lambda x : x*x)(temperature)
        temperature = Lambda(lambda x: tf.Print(x, [x], "temperature = "))(temperature)
        logits = Lambda(lambda x : x[:,1:])(x)
        logits = Lambda(lambda x : tf.Print(x, [x], "logits = "))(logits)
        soft_logits = Multiply()([logits, temperature])
        soft_logits = Lambda(lambda x : tf.Print(x, [x], "soft logits = "))(soft_logits)
    predictions = Activation('softmax')(soft_logits)
    model = Model(inputs = inputs, outputs=predictions)
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def scheduler(epoch):
    if epoch <=15:
        return 1.
    if epoch <= 45:
        return 0.5
    if epoch <= 90:
        return 0.1
    if epoch <= 120:
        return 0.01
    if epoch <= 150:
        return 0.001
    if epoch <= 180:
        return 0.0001
    if epoch <= 210:
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
    logInit(log_filepath + "/log.log")

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
        model = build_model(n=N, num_classes = num_classes)
        print(model.summary())
        # model.load_weights('/home/khoi/NN_calibration_results/')

        # set callback
        change_lr = LearningRateScheduler(scheduler)
        tensorboard = TensorBoard(log_filepath, histogram_freq=1, write_graph=True, write_images=False)
        cbks = [change_lr, tensorboard]

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
        model.save(log_filepath + id + '_' + str(i) + '_' + 'lenet_c10.h5')

        print("Get test accuracy:")
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("Test: accuracy1 = %f  ;  loss1 = %f" % (accuracy, loss))

        print("Pickle models history")
        with open(log_filepath + id + '_' + str(i) + '_' + 'hist_lenet_c10.p', 'wb') as f:
            pickle.dump(hist.history, f)

        K.clear_session()
