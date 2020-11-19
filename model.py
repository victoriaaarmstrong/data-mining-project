from keras.models import Sequential
from keras.layers import Dense

def buildModel():
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

def trainModel(trainX, trainY):

    model = buildModel()

    print('Shape of trainFeatures:' + str(trainX.shape))
    print('Shape of trainLabels:' + str(trainY.shape))

    model.fit(x=trainX,
              y=trainY,
              validation_split=0.15,
              epochs=100,
              batch_size=5,
              verbose=1)

    return model

def testModel(model, testX, testY):
    scores = model.evaluate(testX, testY)
    print('\n')
    print('accuracy: ' + str(scores[1]))

    return