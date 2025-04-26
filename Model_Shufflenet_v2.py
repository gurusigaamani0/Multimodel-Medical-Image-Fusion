# Source code: https://github.com/opconty/keras-shufflenetV2/blob/master/shufflenetv2.py

# Link = https://www.kaggle.com/code/dcosmin/shufflenet-with-keras


# from keras.utils import plot_model
from keras_applications.imagenet_utils import _obtain_input_shape
# from keras.engine.topology import get_source_inputs
# from keras.layers import Input, Conv2D, MaxPool2D, GlobalMaxPooling2D, GlobalAveragePooling2D
# from keras.layers import Activation, Dense
# from keras.models import Model
# import keras.backend as K

import os
import cv2 as cv
from keras import backend as K
from keras.models import Model
# from keras.engine.topology import get_source_inputs
from keras.layers import Activation, Add, Concatenate, Conv2D, GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D, Input, Dense
from keras.layers import MaxPool2D, AveragePooling2D, BatchNormalization, Lambda, DepthwiseConv2D
import numpy as np

from Evaluation import evaluation


def channel_split(x, name=''):
    # equipartition
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 2
    c_hat = Lambda(lambda z: z[:, :, :, 0:ip], name='%s/sp%d_slice' % (name, 0))(x)
    c = Lambda(lambda z: z[:, :, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
    return c_hat, c


def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0, 1, 2, 4, 3))
    x = K.reshape(x, [-1, height, width, channels])
    return x


def shuffle_unit(inputs, out_channels, bottleneck_ratio, strides=2, stage=1, block=1):
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        raise ValueError('Only channels last supported')

    prefix = 'stage{}/block{}'.format(stage, block)
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    if strides < 2:
        c_hat, c = channel_split(inputs, '{}/spl'.format(prefix))
        inputs = c

    x = Conv2D(bottleneck_channels, kernel_size=(1, 1), strides=1, padding='same', name='{}/1x1conv_1'.format(prefix))(
        inputs)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_1'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_1'.format(prefix))(x)
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', name='{}/3x3dwconv'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv'.format(prefix))(x)
    x = Conv2D(bottleneck_channels, kernel_size=1, strides=1, padding='same', name='{}/1x1conv_2'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_2'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_2'.format(prefix))(x)

    if strides < 2:
        ret = Concatenate(axis=bn_axis, name='{}/concat_1'.format(prefix))([x, c_hat])
    else:
        s2 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', name='{}/3x3dwconv_2'.format(prefix))(inputs)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv_2'.format(prefix))(s2)
        s2 = Conv2D(bottleneck_channels, kernel_size=1, strides=1, padding='same', name='{}/1x1_conv_3'.format(prefix))(
            s2)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_3'.format(prefix))(s2)
        s2 = Activation('relu', name='{}/relu_1x1conv_3'.format(prefix))(s2)
        ret = Concatenate(axis=bn_axis, name='{}/concat_2'.format(prefix))([x, s2])

    ret = Lambda(channel_shuffle, name='{}/channel_shuffle'.format(prefix))(ret)

    return ret


def block(x, channel_map, bottleneck_ratio, repeat=1, stage=1):
    x = shuffle_unit(x, out_channels=channel_map[stage - 1],
                     strides=2, bottleneck_ratio=bottleneck_ratio, stage=stage, block=1)

    for i in range(1, repeat + 1):
        x = shuffle_unit(x, out_channels=channel_map[stage - 1], strides=1,
                         bottleneck_ratio=bottleneck_ratio, stage=stage, block=(1 + i))

    return x


def ShuffleNetV2(include_top=True,
                 input_tensor=None,
                 scale_factor=1.0,
                 pooling='max',
                 input_shape=(32, 32, 3),
                 load_model=None,
                 num_shuffle_units=[3, 7, 3],
                 bottleneck_ratio=1,
                 classes=131):
    if K.backend() != 'tensorflow':
        raise RuntimeError('Only tensorflow supported for now')
    name = 'ShuffleNetV2_{}_{}_{}'.format(scale_factor, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))
    input_shape = _obtain_input_shape(input_shape, default_size=224, min_size=28, require_flatten=include_top,
                                      data_format=K.image_data_format())
    out_dim_stage_two = {0.5: 48, 1: 116, 1.5: 176, 2: 244}

    if pooling not in ['max', 'avg']:
        raise ValueError('Invalid value for pooling')
    if not (float(scale_factor) * 4).is_integer():
        raise ValueError('Invalid value for scale_factor, should be x over 4')
    exp = np.insert(np.arange(len(num_shuffle_units), dtype=np.float32), 0, 0)  # [0., 0., 1., 2.]
    out_channels_in_stage = 2 ** exp
    out_channels_in_stage *= out_dim_stage_two[bottleneck_ratio]  # calculate output channels for each stage
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # create shufflenet architecture
    x = Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2),
               activation='relu', name='conv1')(img_input)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)

    # create stages containing shufflenet units beginning at stage 2
    for stage in range(len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = block(x, out_channels_in_stage,
                  repeat=repeat,
                  bottleneck_ratio=bottleneck_ratio,
                  stage=stage + 2)

    if bottleneck_ratio < 2:
        k = 1024
    else:
        k = 2048
    x = Conv2D(k, kernel_size=1, padding='same', strides=1, name='1x1conv5_out', activation='relu')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name='global_max_pool')(x)

    if include_top:
        x = Dense(classes, name='fc')(x)
        x = Activation('softmax', name='softmax')(x)

    if input_tensor:
        inputs = input_tensor
    #     inputs = get_source_inputs(input_tensor)

    else:
        inputs = img_input
    model = Model(inputs, x, name=name)

    if load_model:
        model.load_weights('', by_name=True)
    return model


def Model_shufflenet(train_data, train_target, test_data, test_target, sol=None):
    if sol is None:
        sol = [15, 30, 2]
    activation = ['Relu', 'linear', 'tanh', 'sigmoid']
    optimizer = ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad']

    IMG_SIZE = [32, 32, 3]
    Train_x = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        Train_x[i, :] = cv.resize(train_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    Train_x = Train_x.reshape(Train_x.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Test_x = np.zeros((test_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(test_data.shape[0]):
        Test_x[i, :] = cv.resize(test_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    Test_x = Test_x.reshape(Test_x.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    model = ShuffleNetV2(include_top=True, input_shape=(32, 32, 3), load_model=None,
                         classes=train_target.shape[1])  # 131)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(Train_x, train_target, batch_size=int(sol[0]), epochs=int(sol[1]), verbose=1, validation_data=(Test_x, test_target), callbacks=None)
    Pred = model.predict(test_data)
    Eval = evaluation(Pred, test_target)
    return Eval, Pred
