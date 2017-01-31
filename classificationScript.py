import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import scipy.stats as stats
import sys

#Input taken from commandline to predict one of the characteristics: 0 for size prediction, 1 for intelligence prediction
#cateNo will store either 0 or 1 acccording to user's input
if __name__ == "__main__":
    cateNo = int(sys.argv[1])#input 0 or 1 for 'isBig' or 'isSmart'

#String to store the name of training file which has size data of animals.
size_training_data_file = "sizetrainingdata.csv"

#String to store the name of training file which has intelligence data of animals.
intelligence_training_data_file = "intelligencetrainingdata.csv"

#String to store the name of test file which has the test data.
test_data_file="testdata.csv"

#Array created for two options for prediction: Size or Intelligence.
categories = ['isBig','isSmart']

#training_data_file is string which will store the name of training file according to user choice (cateNo).
training_data_file = size_training_data_file if cateNo==0 else intelligence_training_data_file

#Read the training data file with delimiter "," because CSV stores data as comma-seperated. pd (pandas library) has read_csv as a module for reading CSV files.
#And, drop (delete) the 'animal' column as we don't need it for the algorithm.
#train is of DataFrame format
train = pd.read_csv(training_data_file,sep=",").drop('animal', axis=1)

#loc extracts the data on the basis of location. Basically, in this case it will extract rows according to the condition given
#pos is the data of positive responses. First, the rows which has 'isBig'/'isSmart' as 1 is extracted. 
#Then, the training label column ('isBig' / 'isSmart') is removed.
pos = train.loc[train[categories[cateNo]]==1].drop(categories[cateNo],axis=1)

#pos is the data of negative responses. First, the rows which has 'isBig'/'isSmart' as 0 is extracted.
#Then, the training label column ('isBig' / 'isSmart') is removed.
neg = train.loc[train[categories[cateNo]]==0].drop(categories[cateNo],axis=1)

#Converting to matrix form.
posMat = pos.as_matrix()
negMat = neg.as_matrix()

#Calculating transpose.
posMatT = np.transpose(posMat)
netMatT = np.transpose(negMat)

#Calculate covariance matrix
covPos = np.cov(posMatT)
covNeg = np.cov(posMatT)

#Calculating mean along column
meanPos = np.mean(posMat,axis=0)
meanNeg = np.mean(negMat,axis=0)

#Reading the test data file
testdata = pd.read_csv(test_data_file, sep=',')

#Read the categories[cateNo], i.e, the training label from the test data file and save it as a matrix form.
testlabels = testdata[categories[cateNo]].as_matrix()

#First column has animal name, second column has 'isBig' label and third column has 'isSmall' label, which we don't need to predict the values.
#So, we need to remove the first(0) column and second(1) column from training file.
testdata.drop(testdata.columns[[0,1,2]], axis=1, inplace=True)

#Converting to matrix form
testMat = testdata.as_matrix()

#Calculating probability of positive responses.
#allow_singular=True means allow the covariance matrices which are singular.
posProb = stats.multivariate_normal.pdf(testMat,meanPos,covPos,allow_singular=True)

#Calculating probability of negative responses.
negProb = stats.multivariate_normal.pdf(testMat,meanNeg,covPos,allow_singular=True)

#Prediction matrix
predict = (posProb > negProb)*1

correctOnes = (np.array(testlabels)==np.array(predict))*1
accuracy = np.sum(correctOnes) / float(len(correctOnes))

print "test labels:{}".format(testlabels)
print "predicts:{}".format(predict)
print "correctOnes (1 means correct):{}".format(correctOnes)
print "accuracy:{}".format(accuracy)

#this part calculates the correlation between the output probability and human rating
from scipy.stats.stats import pearsonr
testscores = pd.read_csv("testdatascore.csv")
if cateNo == 0:
	labelScore = testscores['size'].as_matrix()
else:
	labelScore = testscores['intel'].as_matrix()
predictScore = posProb/(posProb+negProb)
print "labelScore: {}".format(labelScore)
print "categorization score: {}".format(predictScore)
print "correlation: {}".format(pearsonr(predictScore,labelScore)[0])

