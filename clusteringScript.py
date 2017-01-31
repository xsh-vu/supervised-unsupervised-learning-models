import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import scipy.stats as stats
import sys

#Input taken from commandline to predict one of the characteristics: 0 for size prediction, 1 for intelligence prediction
#cateNo will store either 0 or 1 acccording to user's input
if __name__ == "__main__":
    cateNo = int(sys.argv[1]) #input 0 or 1 for 'isBig' or 'isSmart'.

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
X = pd.read_csv(training_data_file,sep=",")

#Read the categories[cateNo], i.e, the training label from the training data file and save it as a matrix form. 
#The training label is either 'isBig' or 'isSmart', decided according to user's choice (cateNo).
#Note: 'X' is not a matrix type. It is of DataFrame type. So, we use as_matrix() module provided by pandas to return the matrix form of data.
trainingLabels = X[categories[cateNo]].as_matrix()

#We extracted the training label before. Now, we only need data (observations) according to which clusters will be formed.
#First column has animal name and second column has training label, which we don't need to form clusters.
#So, we need to remove the first(0) column and second(1) column from training file.
#axis = 1 imples removing column. implace = true implies deleting the columns without having to reassign the modified X
X.drop(X.columns[[0,1]], axis=1, inplace=True)

#Since the modified X is also DataFrame format. We need to use as_matrix() to return the matrix form.
trainMat = X.as_matrix()

#Applying K-Means clustering using trainMat data. 2 clusters will be formed.
#By default the number of iterations id 300.
km = KMeans(n_clusters=2, random_state=0).fit(trainMat)

#Reading the test data file
Y = pd.read_csv(test_data_file, sep=",")

#Read the categories[cateNo], i.e, the training label from the test data file and save it as a matrix form. 'Same as before'
testlabels = Y[categories[cateNo]].as_matrix()

#First column has animal name, second column has 'isBig' label and third column has 'isSmart' label, which we don't need to predict the values.
#So, we need to remove the first(0) column and second(1) column from training file.
Y.drop(Y.columns[[0,1,2]], axis=1, inplace=True)

#Converting to matrix form
testMat = Y.as_matrix()

#Predicting the value (either the size or the intelligence)
predict = km.predict(testMat)

#these two lines let us know which cluster represents which label
#and it turned out that cluster 1 represent label 0
print "labels on training data:{}".format(trainingLabels)
print "raw predict outputs on training data:{}".format(km.predict(trainMat))
print "therefore output 0 reprents label 1, and output 1 represents label 0"

# flip the 0 and 1 in predict array
predict = (np.array(predict) == np.zeros(len(predict)))*1
print "correct labels:{}".format(testlabels)
print "predict labels using clustering:{}".format(predict)
correctOnes = (np.array(testlabels)==np.array(predict))*1

# result
print "correct ones(1 is correct, 0 is false):{}".format(correctOnes)
accuracy = np.sum(correctOnes) / float(len(correctOnes))
print "accuracy:{}".format(accuracy)

# calculate the correlation
from scipy.stats.stats import pearsonr
testscores = pd.read_csv("testdatascore.csv")
if cateNo == 0:
    labelScore = testscores['size'].as_matrix()
else:
    labelScore = testscores['intel'].as_matrix()
distances = km.transform(testMat)
disScore = [dis[1]/(dis[0]+dis[1]) for dis in distances] #the closer to cluster0/label1/smart/big, the bigger the score
print "labelScore: {}".format(labelScore)
print "categorization score: {}".format(disScore)
print "correlation: {}".format(pearsonr(disScore,labelScore)[0]) 


