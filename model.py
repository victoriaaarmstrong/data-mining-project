import tensorflow as tf


from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

from tensorflow import keras

from tensorflow.keras import layers, Model
from tensorflow.keras.metrics import MeanSquaredError


def feedForward():
    model = Sequential()
    model.add(Dense(units=128,
                    input_dim=9,
                    activation='softmax'))
    model.add(Dense(units=128,
                    activation='softmax'))
    model.add(Dense(units=3,
                    activation='softmax'))

    print(model.summary())

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])

    return model

def trainModel(model, trainX, trainY):

    print('Shape of trainFeatures:' + str(trainX.shape))
    print('Shape of trainLabels:' + str(trainY.shape))

    model.fit(x=trainX,
              y=trainY,
              validation_split=0.15,
              epochs=100,
              batch_size=5,
              verbose=1)

    ## !! Export model weights to a JSON?

    return model

def testModel(model, testX, testY):
    scores = model.evaluate(testX, testY)

    ## !! Add other metrics
    print('\n')
    print('\t accuracy: ' + str(scores[1]))

    return

