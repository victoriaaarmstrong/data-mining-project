import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import normalize, MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from model import simpleLSTM


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
                     title=title,
                     xlabel='Time',
                     legend=True,
                     xticks=[],
                     subplots=True)

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
        d = {'rest': 0, 'no stress': 0, 'time pressure': 1, 'interruption': 1}
    else:
        d = {'rest': 0, 'no stress': 1, 'time pressure': 2, 'interruption': 3}

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


def timeSeries(data, windowSize, predictionSize):
    min_max_scaler = MinMaxScaler()

    # x_scaled = min_max_scaler.fit_transform(x)

    feature1 = data['HR'].values.reshape(-1, 1)
    feature1 = min_max_scaler.fit_transform(feature1)

    feature2 = data['RMSSD'].values.reshape(-1, 1)
    feature2 = min_max_scaler.fit_transform(feature2)

    feature3 = data['SCL'].values.reshape(-1, 1)
    feature3 = min_max_scaler.fit_transform(feature3)

    x = []
    y = []

    start = 0
    endC = windowSize
    endT = windowSize + predictionSize
    dataLength = len(data.index)

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


def main():
    """
    Returns a train, test and validation set of time series data
    :param windowSize: how big you want the sliding window to be
    :param trainSize: size of training set, default 0.7
    :param testSize: size of testing set, default 0.15
    :param validationSize: side of validation set, default 0.15
    :return:
    """

    WINDOW = 10
    PREDICTION = 10

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

    data = pd.concat(
        [p1, p2, p3, p4, p5, p6, p7, p9, p10, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25],
        axis=0)
    data = data.drop(['Unnamed: 0', 'C', 'timestamp', 'date', 'PP', 'ElapsedTime', 'Condition'], axis=1)
    data = data[~data['label'].isin(['rest'])]
    data = data.replace({'no stress': 0, 'interruption': 1, 'time pressure': 2})

    ###############################################################
    ## Use to just test on whichever three participants you want ##
    ###############################################################
    """
    train = data[~data['subject'].isin(['p2', 'p4', 'p6'])] 
    test = data[data['subject'].isin(['p2', 'p4', 'p6'])]
    
    trainX, trainY = timeSeries(train, WINDOW, PREDICTION)
    testX, testY = timeSeries(test, WINDOW, PREDICTION)
    
    simpleLSTM(WINDOW, PREDICTION, trainX, trainY, testX, testY)
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

        prediction = simpleLSTM(WINDOW, PREDICTION, trainX, trainY, testX, testY)

        mses.append(prediction)

    meses = np.array(mses)
    print(mses.mean())

    return

main()