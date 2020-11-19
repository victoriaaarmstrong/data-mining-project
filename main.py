from dataProcessing import generateData
from model import trainModel, testModel

trainFeatures, trainLabels, testFeatures, testLabels = generateData(3)
model = trainModel('simple', trainFeatures, trainLabels)
testModel(model, testFeatures, testLabels)