import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np

from Evaluation import evaluation


def dilated_residual_block(x, filters, dilation_rate):
    shortcut = x

    x = layers.Conv2D(filters, (3, 3), padding='same', dilation_rate=dilation_rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (3, 3), padding='same', dilation_rate=dilation_rate)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x


def ADRNet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Initial Convolutional Block
    x = layers.Conv2D(64, (7, 7), padding='same', strides=2)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # Dilated Residual Blocks
    x = dilated_residual_block(x, 64, dilation_rate=1)
    x = dilated_residual_block(x, 64, dilation_rate=2)
    x = dilated_residual_block(x, 64, dilation_rate=4)
    x = dilated_residual_block(x, 64, dilation_rate=8)

    # Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    return model


def Model_ADRNet(train_data, train_target, test_data, test_target, sol = None):
    if sol is None:
        sol = [5]
    input_shape = (224, 224, 3)
    num_classes = test_target.shape[-1]

    Feat1 = np.zeros((train_data.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(train_data.shape[0]):
        Feat1[i, :] = np.resize(train_data[i], (input_shape[0], input_shape[1], input_shape[2]))
    train_data = Feat1.reshape(Feat1.shape[0], input_shape[0], input_shape[1], input_shape[2])

    Feat2 = np.zeros((test_data.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(test_data.shape[0]):
        Feat2[i, :] = np.resize(test_data[i], (input_shape[0], input_shape[1], input_shape[2]))
    test_data = Feat2.reshape(Feat2.shape[0], input_shape[0], input_shape[1], input_shape[2])

    model = ADRNet(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(train_data, train_target, epochs=5, batch_size=4, validation_data=(test_data, test_target))
    pred = model.predict(test_data)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, test_target)
    return Eval, pred



if __name__ == '__main__':
    a = np.random.randint(0, 255, (10, 256, 256, 3))
    b = np.random.randint(0, 2, (10, 3))
    c = np.random.randint(0, 255, (10, 256, 256, 3))
    d = np.random.randint(0, 2, (10, 3))
    Eval, pred =Model_ADRNet(a, b, c, d)
    nh = 5645