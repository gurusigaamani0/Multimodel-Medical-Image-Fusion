from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D
from keras.models import Model
import numpy as np

def res_block(input_layer, filters, kernel_size=(3, 3), padding='same'):
    x = Conv2D(filters, kernel_size, padding=padding, activation='relu')(input_layer)
    x = Conv2D(filters, kernel_size, padding=padding)(x)
    x = concatenate([input_layer, x], axis=-1)
    x = Conv2D(filters, kernel_size, padding=padding, activation='relu')(x)
    return x

def res_unet(input_shape, num_classes):
    inputs = Input(input_shape)

    # Contracting path
    conv1 = res_block(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = res_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = res_block(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = res_block(pool3, 512)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = res_block(pool4, 1024)

    # Expansive path
    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = concatenate([up6, conv4], axis=-1)
    conv6 = res_block(up6, 512)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = concatenate([up7, conv3], axis=-1)
    conv7 = res_block(up7, 256)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = concatenate([up8, conv2], axis=-1)
    conv8 = res_block(up8, 128)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = concatenate([up9, conv1], axis=-1)
    conv9 = res_block(up9, 64)

    # Output layer
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)

    model = Model(inputs, outputs)
    return model


def Model_Res_Unet(Data, tar):
    # Example usage
    input_shape = (256, 256, 3)
    num_classes = 3  # Replace with the number of output classes
    Target = np.zeros((tar.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(tar.shape[0]):
        temp_1 = np.resize(tar[i], (input_shape[0] * input_shape[1], input_shape[2]))
        Target[i] = np.reshape(temp_1, (input_shape[0], input_shape[1], input_shape[2]))
    model = res_unet(input_shape, num_classes)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(Data, Target, epochs=5, batch_size=2, steps_per_epoch=5)
    # Predict using the trained model
    predicted_images = model.predict(Data)
    return predicted_images
