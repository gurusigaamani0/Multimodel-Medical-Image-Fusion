import keras
import numpy as np
import cv2 as cv
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, Dropout, GlobalAveragePooling2D, \
    BatchNormalization, concatenate, AveragePooling2D

from Evaluation import evaluation

def conv_layer(conv_x, filters):
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
    conv_x = Conv2D(filters, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(conv_x)
    conv_x = Dropout(0.2)(conv_x)

    return conv_x


def dense_block(block_x, filters, growth_rate, layers_in_block):
    for i in range(layers_in_block):
        each_layer = conv_layer(block_x, growth_rate)
        block_x = concatenate([block_x, each_layer], axis=-1)
        filters += growth_rate

    return block_x, filters


def transition_block(trans_x, tran_filters):
    trans_x = BatchNormalization()(trans_x)
    trans_x = Activation('relu')(trans_x)
    trans_x = Conv2D(tran_filters, (1, 1), kernel_initializer='he_uniform', padding='same', use_bias=False)(trans_x)
    trans_x = AveragePooling2D((2, 2), strides=(2, 2))(trans_x)

    return trans_x, tran_filters


def dense_net(num_of_class=1):
    dense_block_size = 3
    layers_in_block = 4
    growth_rate = 12
    filters = growth_rate * 2
    input_img = Input(shape=(32, 32, 3))
    x = Conv2D(24, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(input_img)

    dense_x = BatchNormalization()(x)
    dense_x = Activation('relu')(x)

    dense_x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(dense_x)
    for block in range(dense_block_size - 1):
        dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
        dense_x, filters = transition_block(dense_x, filters)

    dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
    dense_x = BatchNormalization()(dense_x)
    dense_x = Activation('relu')(dense_x)
    dense_x = GlobalAveragePooling2D()(dense_x)

    output = Dense(num_of_class, activation='softmax')(dense_x)
    model = Model(input_img, output)

    return model


def Model_DenseNet(train_data, Target, test_data, test_tar, Batch_size=None, sol=None):

    if Batch_size is None:
        Batch_size = 4
    if sol is None:
        sol = 5
    IMG_SIZE = [32, 32, 3]
    Train_x = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        Train_x[i, :] = cv.resize(train_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    train_data = Train_x.reshape(Train_x.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Test_x = np.zeros((test_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(test_data.shape[0]):
        Test_x[i, :] = cv.resize(test_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    test_data = Test_x.reshape(Test_x.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    model = dense_net(Target.shape[1])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, Target, steps_per_epoch=10, batch_size=Batch_size, epochs=int(sol), validation_data=(test_data, test_tar) )
    pred = model.predict(test_data)
    Eval = evaluation(pred, test_tar)
    return Eval, pred


if __name__ == '__main__':
    a = np.random.uniform(0, 255, (10, 256,256, 3))
    b = np.random.randint(0, 2, 10)
    c = np.random.uniform(0, 255, (10, 256,256, 3))
    d = np.random.randint(0, 2, 10)
    eval, pred = Model_DenseNet(a, b, c, d)

