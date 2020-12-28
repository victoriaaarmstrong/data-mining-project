import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, LSTM, Reshape
from tensorflow.keras.models import Sequential

def meanImputation(data):
    """
    :param data: the DataFrame object for any participant
    :return: the DataFrame object that has been filled
    """

    data['HR'].fillna(data['HR'].mean(), inplace=True)
    data['RMSSD'].fillna(data['RMSSD'].mean(), inplace=True)
    data['SCL'].fillna(data['SCL'].mean(), inplace=True)

    return data


def countMissing(participant, fileName):
    """
    Prints metrics for missing data for any given participant set
    :param participant: participant DataFrame
    :return: None
    """

    ## Initialize counters
    numHR = 0
    numRMSSD = 0
    numSCL = 0
    size = len(participant.index)

    ## Iterates through DF checking for NaN values
    for index, row in participant.iterrows():
        if np.isnan(row['HR']) == True:
            numHR += 1
        if np.isnan(row['RMSSD']) == True:
            numRMSSD += 1
        if np.isnan(row['SCL']) == True:
            numSCL += 1

    ## Calculates percentages
    perHR = numHR / size
    perRMSSD = numRMSSD / size
    perSCL = numSCL / size

    ## Displays the percentages
    print("Missing data percentages for " + fileName + ":")
    print(" \t HR missing: " + str(perHR))
    print(" \t RMSSD missing: " + str(perRMSSD))
    print(" \t SCL missing: " + str(perSCL))

    return


def importCSV(fileName):
    """
    :param fileName: the name of the file to read in
    :return: DataFrame object
    """

    ## Read in the csv file to DataFrame
    participant = pd.read_csv(fileName)

    ## Fill missing data
    #participant['HR'] = participant['HR'].fillna(0) ## zero-insertion
    participant = meanImputation(participant)

    participant = participant.drop(['Unnamed: 0', 'C', 'timestamp', 'date', 'PP', 'ElapsedTime', 'Condition'], axis=1)
    participant = participant[~participant['label'].isin(['rest'])]
    participant = participant.replace({'no stress': 0, 'interruption': 1, 'time pressure': 2})

    ## To display missing metrics if you want
    ##countMissing(participant, fileName)

    return participant


def plotFeature(participant, title):
    """
    :param participant: the DataFrame object for any participant
    :param title: graph title
    :return: None
    """
    participant.plot(x='timestamp',
                     y=['HR'], #, 'RMSSD', 'SCL'],
                     kind='line',
                     title=title,
                     xlabel='Time',
                     legend=True,
                     xticks=[],
                     subplots=True)

    plt.show()

    return


def timeSeries(data, windowSize, predictionSize):
    #min_max_scaler = MinMaxScaler()

    vals = data['subject'].unique()

    x = []
    y = []

    for v in vals:
        participant = data.loc[data['subject'] == str(v)]

        feature1 = participant['HR'].values.reshape(-1, 1)
        # feature1 = min_max_scaler.fit_transform(feature1) ## where it had been normalized

        feature2 = participant['RMSSD'].values.reshape(-1, 1)
        # feature2 = min_max_scaler.fit_transform(feature2) ## where it had been normalized

        feature3 = participant['SCL'].values.reshape(-1, 1)
        # feature3 = min_max_scaler.fit_transform(feature3) ## where it had been normalized

        start = 0
        endC = windowSize
        endT = windowSize + predictionSize
        dataLength = len(participant.index)

        while endT <= dataLength:
            t1 = feature1[start:endC]
            t2 = feature2[start:endC]
            t3 = feature2[start:endC]
            context = np.concatenate((t1, t2, t3), axis=1)

            t4 = feature1[endC:endT]
            t5 = feature2[endC:endT]
            t6 = feature3[endC:endT]
            target = np.concatenate((t4, t5, t6), axis=1)

            x.append(context)
            y.append(target)

            start += 1
            endC += 1
            endT += 1

    x = np.array(x)
    y = np.array(y)

    return x, y


def lstmTrain(windowLength, predictionSize, trainX, trainY):

    model = Sequential()
    model.add(LSTM(32, input_shape=(windowLength,3), return_sequences=False))
    model.add(Dense(predictionSize*3, kernel_initializer=tf.initializers.zeros))
    model.add(Reshape([predictionSize, 3]))

    model.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsoluteError(), metrics=[tf.keras.metrics.MeanAbsoluteError()])

    model.fit(x=trainX,
              y=trainY,
              epochs=50,
              batch_size=5)

    return model


def lstmTest(model, testX, testY):

    predictions = model.evaluate(testX, testY)
    outputs = model.predict(testX)
    print('\n accuracy: ')
    print(predictions[1])

    return predictions[1] #outputs


def main():
    ## Read in participant data files
    p1 = importCSV('p1.csv')
    p2 = importCSV('p2.csv')
    p3 = importCSV('p3.csv')
    p4 = importCSV('p4.csv')
    p5 = importCSV('p5.csv')
    p6 = importCSV('p6.csv')
    p7 = importCSV('p7.csv')
    p9 = importCSV('p9.csv')
    p10 = importCSV('p10.csv')
    p12 = importCSV('p12.csv')
    p13 = importCSV('p13.csv')
    p14 = importCSV('p14.csv')
    p15 = importCSV('p15.csv')
    p16 = importCSV('p16.csv')
    p17 = importCSV('p17.csv')
    p18 = importCSV('p18.csv')
    p19 = importCSV('p19.csv')
    p20 = importCSV('p20.csv')
    p21 = importCSV('p21.csv')
    p22 = importCSV('p22.csv')
    p23 = importCSV('p23.csv')
    p24 = importCSV('p24.csv')
    p25 = importCSV('p25.csv')

    ############################################################
    ## Plots participant 21's data in disjunct condition sets ##
    ############################################################
    """
    noStress = p21[~p21['label'].isin(['1','2'])]
    interruptionStress = p21[~p21['label'].isin(['0','2'])]
    pressureStress = p21[~p21['label'].isin(['0','1'])]

    ## Make sure you comment out the drop labels in importCSV or else you get an error plotting with the timestamp
    plt.plot(noStress['timestamp'], noStress['SCL'], c='g', label='neutral')
    plt.plot(interruptionStress['timestamp'], interruptionStress['SCL'], label='interruption')
    plt.plot(pressureStress['timestamp'], pressureStress['SCL'], label='pressure')
    plt.legend(loc='upper left')
    plt.show()

    """

    ## Set window and prediction parameters to play around with
    WINDOW = 10
    PREDICTION = 10


    ## Combine all the participants into one DataFrame
    data = pd.concat([p1, p2, p3, p4, p5, p6, p7, p9, p10, p12,
                      p13, p14, p15, p16, p17, p18, p19,
                      p20, p21, p22, p23, p24, p25],
                     axis=0)

    ######################################################
    ## Within-participant training, testing and gaphing ##
    ######################################################
    """
    ## Train test split of all data
    #train, test = train_test_split(data, test_size=0.3, random_state=42)

    ## Train test split of neutral data
    #data = data[~data['label'].isin(['1', '2'])]
    #train, test = train_test_split(data, test_size=0.3, random_state=42)

    ## Train test split of stress condition data
    #data = data[~data['label'].isin(['0'])]
    #train, test = train_test_split(data, test_size=0.3, random_state=42)

    ## Makes time series data from the desired data set
    trainX, trainY = timeSeries(train, WINDOW, PREDICTION)
    testX, testY = timeSeries(test, WINDOW, PREDICTION)

    ## Trains model
    model = lstmTrain(WINDOW, PREDICTION, trainX, trainY)

    ## Makes predictions and outputs accuracy
    predictions = lstmTest(model, testX, testY)


    ## Initialize for plotting predictions
    input = []
    predicted = []
    true = []

    ## Get values to plot
    for i in range(WINDOW):
        input.append(testX[0][i][1])

    for i in range(PREDICTION):
        predicted.append(predictions[0][i][1])
        true.append(testY[0][i][1])

    ## Setting X-labels for predictions
    x1 = [0,1,2,3,4,5,6,7,8,9]
    x2 = [10,11,12] #,13,14,15,16,17,18,19]

    ## Plot
    plt.plot(x1, input, c='g', label='input' )
    plt.scatter(x2, predicted, c='b', label='predicted')
    plt.scatter(x2, true, c='r', label='true')
    plt.legend(loc='upper left')
    plt.show()
    """

    #######################################################
    ## Unseen participant training, testing and graphing ##
    #######################################################
    """
    #participantList = ['p1', 'p2', 'p3', 'p4']
    #participantList = ['p15', 'p16', 'p17', 'p18']
    participantList = ['p20', 'p21', 'p22', 'p23']
    
    ## Train test split of all data 
    #train = data[~data['subject'].isin(participantList)]
    #test = data[data['subject'].isin(participantList)]

    ## Train test split of neutral data
    #data = data[~data['label'].isin(['1', '2'])]
    #train = data[~data['subject'].isin(participantList)]
    #test = data[data['subject'].isin(participantList)]

    ## Train test split of stress condition data
    data = data[~data['label'].isin(['0'])]
    train = data[~data['subject'].isin(participantList)]
    test = data[data['subject'].isin(participantList)]

    ## Makes time series data from the desired data set
    trainX, trainY = timeSeries(train, WINDOW, PREDICTION)
    testX, testY = timeSeries(test, WINDOW, PREDICTION)

    ## Trains model
    model = lstmTrain(WINDOW, PREDICTION, trainX, trainY)

    ## Makes predictions and outputs accuracy
    predictions = lstmTest(model, testX, testY)

    ## Initialize for plotting predictions
    input = []
    predicted = []
    true = []

    ## Get values to plot
    for i in range(WINDOW):
        input.append(testX[0][i][1])

    for i in range(PREDICTION):
        predicted.append(predictions[0][i][1])
        true.append(testY[0][i][1])

    ## Setting X-labels for predictions
    x1 = [0,1,2,3,4,5,6,7,8,9]
    x2 = [10,11,12,13,14,15,16,17,18,19]
    
    ## Plot
    plt.plot(x1, input, c='g', label='input' )
    plt.scatter(x2, predicted, c='b', label='predicted')
    plt.scatter(x2, true, c='r', label='true')
    plt.legend(loc='upper left')
    plt.show()
    """

    ###############################################################
    ### Use to average a bunch of independent test particiapnts ###
    ###############################################################
    testSplits = [['p1, p2, p3, p4'],
                  ['p2, p3, p4, p5'],
                  ['p3, p4, p5, p6'],
                  ['p4, p5, p6, p7'],
                  ['p5, p6, p7, p9'],
                  ['p6, p7, p9, p10'],
                  ['p7, p9, p10, p12'],
                  ['p9, p10, p12, p13'],
                  ['p10, p12, p13, p14'],
                  ['p12, p13, p14, p15'],
                  ['p13, p14, p15, p16'],
                  ['p14, p15, p16, p17'],
                  ['p15, p16, p17, p18'],
                  ['p16, p17, p18, p19'],
                  ['p17, p18, p19, p20'],
                  ['p18, p19, p20, p21'],
                  ['p19, p20, p21, p22'],
                  ['p20, p21, p22, p23'],
                  ['p21, p22, p23, p24'],
                  ['p22, p23, p24, p25']]

    mses = []

    for testSet in testSplits:
        train = data[~data['subject'].isin(testSet)]
        test = data[data['subject'].isin(testSet)]

        trainX, trainY = timeSeries(train, WINDOW, PREDICTION)
        testX, testY = timeSeries(test, WINDOW, PREDICTION)

        model = lstmTrain(WINDOW, PREDICTION, trainX, trainY)
        prediction = lstmTest(model, testX, testY)
        
        mses.append(prediction)

    mses = np.array(mses)
    print(mses.mean())

    return

main()