import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



def meanImputation(data):
    """
    :param data: the DataFrame object for any participant
    :return: the DataFrame object that has been filled
    """

    data['HR'].fillna(data['HR'].mean(), inplace=True)
    data['RMSSD'].fillna(data['RMSSD'].mean(), inplace=True)
    data['SCL'].fillna(data['SCL'].mean(), inplace=True)

    return data

def countMissing(participant):
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

    ## Drop the unneccessary columns
    ## !! Don't think I actually need to do this because feature extraction is just going to happen in the sliding window
    ## participant = participant.drop(['Condition', 'C', 'Unnamed: 0', 'date', 'subject'], axis = 1)

    ## Fill missing data
    participant = meanImputation(participant)

    ## !! Normalize?
    ## !! Could maybe do this in the tensor - there should be a function for this

    ## To display missing metrics if you want
    ##countMissing(participant)

    return participant


def plotFeature(participant, title):
    """
    :param participant: the DataFrame object for any participant
    :param title: graph title
    :return: None
    """
    participant.plot(x='timestamp',
                     y=['HR', 'RMSSD', 'SCL'],
                     kind='line',
                     title= title,
                     xlabel='Time',
                     legend=True,
                     xticks=[],
                     subplots = True)

    plt.show()

    return

def generateSamples(numTargets, df, windowSize):
    """

    :param numTargets:
    :param df:
    :param lenContext:
    :param lenTarget:
    :return:
    """

    if numTargets == 2:
        d = {'rest':0, 'no stress':0, 'time pressure':1, 'interruption':1}
    else:
        d = {'rest':0, 'no stress':1, 'time pressure':2, 'interruption':3}

    df = df.replace(d)


    ## Initialize variables
    feature1 = df['HR'].tolist()
    feature2 = df['RMSSD'].tolist()
    feature3 = df['SCL'].tolist()
    targets = df['label'].tolist()

    x = []
    y = []

    start = 0
    end = windowSize
    dataLength = len(df.index)

    ## !! Really check that this is iterating through how you want it to, it might not be
    while end < dataLength:
        context = feature1[start:end] + feature2[start:end] + feature3[start:end]
        x.append(context)
        y.append(targets[start:end])
        start += 1
        end += 1

    ## !! Maybe need to reshape y? not sure
    return x, y

def myTimeSeries(df):

    return df


def tfTimeSeries(df):
    data = df[['HR','RMSSD','SCL']]
    targets = df['label']

    data = np.array(data).astype('float32')
    targets = targets.replace({'rest':0, 'no stress':0, 'interruption':1, 'time pressure':1})
    targets = np.array(targets).astype('float32')

    sequence_length = 3
    timeSeries = tf.keras.preprocessing.timeseries_dataset_from_array(
        data, targets, sequence_length, sequence_stride=1, sampling_rate=1,
        batch_size=1, shuffle=True, seed=None, start_index=None, end_index=None
    )

    return timeSeries

def kerasTimeseriesGenerator(df):
    data = df[['HR','RMSSD','SCL']]
    targets = df['label']
    targets = targets.replace({'rest': 0, 'no stress': 0, 'interruption': 1, 'time pressure': 1})
    data_gen = TimeseriesGenerator(data, targets,
                                   length=3, sampling_rate=1)
    batch_0 = data_gen[0]
    train, test = batch_0

    return train, test

def generateData(windowSize):
    """
    Returns a train, test and validation set of time series data
    :param windowSize: how big you want the sliding window to be
    :param trainSize: size of training set, default 0.7
    :param testSize: size of testing set, default 0.15
    :param validationSize: side of validation set, default 0.15
    :return:
    """

    ## Read in participant data files
    p1 = importCSV('p1.csv')
    p2 = importCSV('p2.csv')
    ##p3 = importCSV('p3.csv')
    ##p4 = importCSV('p4.csv')
    ##p5 = importCSV('p5.csv')
    ##p6 = importCSV('p6.csv')
    ##p7 = importCSV('p7.csv')
    ## p8 = importCSV('p8.csv') 100% of HR and RMSSD missing
    p9 = importCSV('p9.csv')
    ##p10 = importCSV('p10.csv')
    ## p11 = importCSV('p11.csv') 100% of HR and RMSSD missing
    ##p12 = importCSV('p12.csv')
    ##p13 = importCSV('p13.csv')
    #3p14 = importCSV('p14.csv')
    ##p15 = importCSV('p15.csv')
    ##p16 = importCSV('p16.csv')
    ##p17 = importCSV('p17.csv')
    ##p18 = importCSV('p18.csv')
    ##p19 = importCSV('p19.csv')
    ##p20 = importCSV('p20.csv')
    ##p21 = importCSV('p21.csv')
    ##p22 = importCSV('p22.csv')
    ##p23 = importCSV('p23.csv')
    ##p24 = importCSV('p24.csv')
    ##p25 = importCSV('p25.csv')

    """
    participantDFs = [p1, p2, p3, p4, p5, p6, p7, p9, p10, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25]
    
    features = []
    labels = []

    for participant in participantDFs:
        temp1, temp2 = generateSamples(2, participant, 3)
        features = features + temp1
        labels = labels + temp2

    trainX, testX, trainY, testY = train_test_split(features, labels, test_size=0.15)

    trainFeatures = np.array(trainX)
    trainLabels = np.array(trainY)
    testFeatures = np.array(testX)
    testLabels = np.array(testY)

    """
    ##data = pd.concat([p1, p2, p3, p4, p5, p6, p7, p9, p10, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25], axis=0)

    trainX, trainY = kerasTimeseriesGenerator(p9)

    """
        b = tfTimeSeries(p1)
    c = tfTimeSeries(p2)

    timeSeries = tf.concat([a,b,c], 0)

    for t in timeSeries:
        print(t)
    
    """
    #trainX = np.reshape(trainX, [])
    print(trainX.shape)
    print(trainY.shape)

    model = Sequential()
    model.add(tf.keras.Input(shape=(None,3,3)))
    model.add(Dense(units=128,
                    activation='softmax'))
    model.add(Dense(units=3,
                    activation='softmax'))

    print(model.summary())

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])

    model.fit(x=trainX,
              y=trainY,
              validation_split=0.15,
              epochs=100,
              batch_size=5,
              verbose=1)

    return ##trainFeatures, trainLabels, testFeatures, testLabels

generateData(3)