import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Add
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention


def residual_block(x, filters):
    shortcut = Conv2D(filters, (1, 1), padding='same')(x)  # Ensure the shortcut has the same shape as the main path
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def transformer_block(x, num_heads, ff_dim):
    input_shape = x.shape[-1]
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=input_shape)(x, x)
    attention_output = LayerNormalization()(attention_output + x)
    ffn = layers.Dense(ff_dim, activation='relu')(attention_output)
    ffn = layers.Dense(input_shape)(ffn)
    return LayerNormalization()(ffn + attention_output)


def attention_gate(x, g, inter_shape):
    # theta_x = Conv2D(inter_shape, (1, 1), strides=(2, 2), padding='same')(x)
    theta_x = Conv2D(inter_shape, (1, 1), padding='same')(x)
    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
    add_xg = layers.add([theta_x, phi_g])
    add_xg = layers.ReLU()(add_xg)
    psi = Conv2D(1, (1, 1), padding='same')(add_xg)
    psi = layers.Activation('sigmoid')(psi)
    upsample_psi = UpSampling2D(size=(2, 2))(psi)
    att_x = layers.multiply([upsample_psi, x])
    return att_x


def encoder_block(x, filters):
    x = residual_block(x, filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p


def decoder_block(x, skip, filters):
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(filters, (2, 2), padding='same')(x)  # Ensure compatibility for concatenation
    x = concatenate([x, skip])
    x = residual_block(x, filters)
    return x


def MA_TRUnetPlusPlus(input_shape, num_classes, num_heads=4, ff_dim=128):
    inputs = Input(shape=input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bottleneck
    b = transformer_block(p4, num_heads, ff_dim)

    # Decoder
    d1 = decoder_block(b, s4, 512)
    d1 = attention_gate(d1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d2 = attention_gate(d2, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d3 = attention_gate(d3, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    d4 = attention_gate(d4, s1, 64)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(d4)

    model = models.Model(inputs, outputs)

    return model


def Model_MA_TRUnet_plusplus(Data, Target):

    input_shape = (256, 256, 3)
    num_classes = 3
    Train_Temp = np.zeros((Data.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(Data.shape[0]):
        Train_Temp[i, :] = np.resize(Data[i], (input_shape[0], input_shape[1], input_shape[2]))
    Train_X = Train_Temp.reshape(Train_Temp.shape[0], input_shape[0], input_shape[1], input_shape[2])

    Test_Temp = np.zeros((Target.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(Target.shape[0]):
        Test_Temp[i, :] = np.resize(Target[i], (input_shape[0], input_shape[1], input_shape[2]))
    Train_Y = Test_Temp.reshape(Test_Temp.shape[0], input_shape[0], input_shape[1], input_shape[2])

    model = MA_TRUnetPlusPlus(input_shape, num_classes)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(Train_X, Train_Y, epochs=10)
    pred = model.predict(Train_X)
    return pred


if __name__ == '__main__':
    a = np.random.randint(0, 255, (10, 256, 256, 3))
    c = np.random.randint(0, 255, (10, 256, 256, 3))
    pred = Model_MA_TRUnet_plusplus(a, c)
    nh = 5645
