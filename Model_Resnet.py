import numpy as np
import cv2 as cv
from keras import Sequential
from keras.applications import ResNet50
from keras.layers import Dense
from Evaluation import evaluation


def Model_RESNET(train_data, train_target, test_data, test_target, Batch_size, Act=None, sol=None):
    if sol is None:
        sol = [1]
    if Act is None:
        Act = 1

    IMG_SIZE = [224, 224, 3]
    Feat1 = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        Feat1[i, :] = np.resize(train_data[i], (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    train_data = Feat1.reshape(Feat1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Feat2 = np.zeros((test_data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(test_data.shape[0]):
        Feat2[i, :] = np.resize(test_data[i], (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    test_data = Feat2.reshape(Feat2.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    base_model = Sequential()
    base_model.add(ResNet50(include_top=False, weights='imagenet', pooling='max'))
    activation = ['Relu', 'linear', 'tanh', 'sigmoid']
    base_model.add(Dense(units=int(sol[0]), activation=activation[int(Act)]))  # 'linear'
    base_model.compile(loss='binary_crossentropy', metrics=['acc'])
    base_model.summary()
    base_model.fit(train_data, train_target, epochs=5, batch_size=Batch_size, validation_data=(test_data, test_target))
    pred = base_model.predict(test_data)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, test_target)
    return Eval, pred

