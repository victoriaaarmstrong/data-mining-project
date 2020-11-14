import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def meanImputation(data):
    """
    :param data: the DataFrame object for any participant
    :return: the DataFrame object that has been filled
    """

    data['HR'].fillna(data['HR'].mean(), inplace=True)
    data['RMSSD'].fillna(data['RMSSD'].mean(), inplace=True)
    data['SCL'].fillna(data['SCL'].mean(), inplace=True)

    return data


def importCSV(fileName):
    """
    :param fileName: the name of the file to read in
    :return: DataFrame object
    """

    ## Read in the csv file to DataFrame
    participant = pd.read_csv(fileName)

    ## Drop the unneccessary columns
    participant = participant.drop(['Condition', 'C', 'Unnamed: 0', 'date', 'subject'], axis = 1)
    ## TO DO
    ## Do I need to keep the Elapsed time? I know they're all one minute average time stamps

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

    participant = meanImputation(participant)

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


def readAllData():
    """
    :return: None
    """

    ## Read in participant data files
    p1 = importCSV('p1.csv')
    p2 = importCSV('p2.csv')
    p3 = importCSV('p3.csv')
    p4 = importCSV('p4.csv')
    p5 = importCSV('p5.csv')
    p6 = importCSV('p6.csv')
    p7 = importCSV('p7.csv')
    ## p8 = importCSV('p8.csv') 100% of HR and RMSSD missing
    p9 = importCSV('p9.csv')
    p10 = importCSV('p10.csv')
    ## p11 = importCSV('p11.csv') 100% of HR and RMSSD missing
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

    return

readAllData()