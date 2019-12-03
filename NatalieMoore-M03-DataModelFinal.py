"""    
Natalie Moore
Assignment L09

This dataset is from the UCI ML archive.
The data is from a 1996 US Census. Not all adults may have participated.
You can find the data at the following URL:
    https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data

Number of observations/attributes: 48842/14
Attributes and their type/distribution:
    age: integer/chi-squared
    workclass: object/exponential
    fnlwgt: integer/log normal
    education: object/hypergeometric
    education-num: integer/weibull
    marital-status: object/gaussian
    occupation: object/gaussian
    relationship: object/gaussian
    race: object/hypergeometric
    sex: object/bernoulli
    capital-gain: integer/uniform
    capital-loss: integer/uniform
    hours-per-week: integer/hypergeometric
    native-country: object/hypergeometric
    salary: object/bernoulli (either <50k or >50k)
    
NOTE: It's worth noting that I dropped the fnlwgt (Final Weight), capital-gain,
and capital-loss attributes from my data set prior to analysis because Final 
Weight was not proven to be useful or meaningful in relation to any other 
attribute, and the capital-gain/loss attributes were majority == 0 after 
removing outliers. I also dropped the NativeCountry column because it's majority
people from the US. Finally, I dropped the Education column because it is basically
the same as the Education(num) column.
    
Below I will first perform K-Means on the Age and Education(num) attributes to 
see the relationship between a US adult's age and how many years of education
they've completed.

After K-Means is applied, I will then use the Nearest Neighbors regression 
model and the Logistic Regression classification model to train and test the 
Census data.

After the classification/regression models are applied, I will then create a 
confusion matrix from my predictions and calculate/present the ROC curve and 
it's AUC using the sklearn package.
"""
#importing necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans as Kmeans
from sklearn.metrics import *
from copy import deepcopy

#figure parameters to make them bigger
plt.rcParams["figure.figsize"] = [10, 10]
matplotlib.rcParams.update({'font.size': 20})

#creating my dataframe called Census from the UCI archive
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
Census = pd.read_csv(url, header=None) 
Census.columns = ["Age","WorkClass","FinalWeight","Education",
                  "Education(num)","MaritalStatus","Occupation",
                  "Relationship","Race","Sex","Capital-Gain",
                  "Capital-Loss","Hours/Week","NativeCountry","Salary"]
Census = Census.drop("FinalWeight", axis = 1) #dropping FinalWeight column because it's not relavent/useful 
#below are removed b/c outlier values revealed the only non-outliers == 0
Census = Census.drop("Capital-Gain", axis = 1) #dropping Capital-Gain column because it's mostly 0
Census = Census.drop("Capital-Loss", axis = 1) #dropping Capital-Loss column because it's mostly 0
Census = Census.drop("NativeCountry", axis = 1) #dropping NativeCountry column because it's mostly USA-born people
Census = Census.drop("Education", axis = 1) #dropping Education column because it's the same thing as Education(num)

#imputing missing values onto most frequent value
Census.loc[Census.loc[:,"WorkClass"] == " ?", "WorkClass"] = " Private"
Census.loc[Census.loc[:,"Occupation"] == " ?", "Occupation"] = " Prof-specialty"

#de-coding the Education column
Census.loc[Census.loc[:,"Sex"] == " Male", "Sex"] = 0
Census.loc[Census.loc[:,"Sex"] == " Female", "Sex"] = 1
#renaming Education column to be Education_decoded
Census.rename(columns={'Sex': 'Sex_decoded'}, inplace=True)

#de-coding the Race column
Census.loc[Census.loc[:,"Salary"] == " <=50K", "Salary"] = 0
Census.loc[Census.loc[:,"Salary"] == " >50K", "Salary"] = 1
#renaming Race column to be Race_decoded
Census.rename(columns={'Salary': 'Salary_decoded'}, inplace=True)

#de-coding the WorkClass column
Census.loc[Census.loc[:,"WorkClass"] == " Federal-gov", "WorkClass"] = 3 #government job
Census.loc[Census.loc[:,"WorkClass"] == " Local-gov", "WorkClass"] = 3 #government job
Census.loc[Census.loc[:,"WorkClass"] == " State-gov", "WorkClass"] = 3 #government job
Census.loc[Census.loc[:,"WorkClass"] == " Private", "WorkClass"] = 1
Census.loc[Census.loc[:,"WorkClass"] == " Self-emp-inc", "WorkClass"] = 2 #self-employed job
Census.loc[Census.loc[:,"WorkClass"] == " Self-emp-not-inc", "WorkClass"] = 2 #self-employed job
Census.loc[Census.loc[:,"WorkClass"] == " Never-worked", "WorkClass"] = 0 #no job
Census.loc[Census.loc[:,"WorkClass"] == " Without-pay", "WorkClass"] = 0 #no job
#renaming Race column to be Race_decoded
Census.rename(columns={'WorkClass': 'WorkClass_decoded'}, inplace=True)

#normalizing continuous/integer-based Census columns
Census.loc[:,"Age"] = (Census.loc[:,"Age"]-min(Census.loc[:,"Age"]))/(max(Census.loc[:,"Age"])-min(Census.loc[:,"Age"]))
Census.loc[:,"Hours/Week"] = (Census.loc[:,"Hours/Week"]-min(Census.loc[:,"Hours/Week"]))/(max(Census.loc[:,"Hours/Week"])-min(Census.loc[:,"Hours/Week"]))
Census.loc[:,"Education(num)"] = (Census.loc[:,"Education(num)"]-min(Census.loc[:,"Education(num)"]))/(max(Census.loc[:,"Education(num)"])-min(Census.loc[:,"Education(num)"]))
#renaming the normalized columns
Census.rename(columns={'Age': 'Age_norm'}, inplace=True)
Census.rename(columns={'Hours/Week': 'Hours_norm'}, inplace=True)
Census.rename(columns={'Education(num)': 'Education(num)_norm'}, inplace=True)

#turning off the Pandas warning I get with my decode function below
pd.set_option('mode.chained_assignment', None)

#decode function to decode the remaining class columns for ML model
def Decode(data): 
    N = len(data)
    x = np.unique(data)
    Nx = len(x)
    code = np.arange(0, len(x), 1)
    
    for i in range(N):
        for j in range(Nx):
            if data.iloc[i] == x[j]:
                data.iloc[i] = code[j]
    return data

#decoding class columns
Decode(Census.loc[:,"MaritalStatus"])
Decode(Census.loc[:,"Occupation"])
Decode(Census.loc[:,"Relationship"])
Decode(Census.loc[:,"Race"])
#renaming decoded class columns 
Census.rename(columns={'MaritalStatus': 'MaritalStatus_decoded'}, inplace=True)
Census.rename(columns={'Occupation': 'Occupation_decoded'}, inplace=True)
Census.rename(columns={'Relationship': 'Relationship_decoded'}, inplace=True)
Census.rename(columns={'Race': 'Race_decoded'}, inplace=True)

#######################################
"""K-Means"""
#creating a newdataframe of Age (normalized) and Education_num (normalized) 
#to use for Kmeans
Variables4Kmeans = pd.concat([Census.loc[:,"Age_norm"], Census.loc[:,"Education(num)_norm"]], axis = 1)

#using the Kmeans package to fit the data to 2 data clusters
#using the Kmeans package to find the cluster coordinates
kmeans = Kmeans(n_clusters=2).fit(Variables4Kmeans)
centroids = kmeans.cluster_centers_
print('Centroid coordinates Figure 1:')
print(centroids)

#plotting the normalized Age and Education_num attributes against each other
#the colors of the points are determined by the Kmeans package (meaning each point is assigned a label from Kmeans)
plt.scatter(Variables4Kmeans['Age_norm'],Variables4Kmeans['Education(num)_norm'], c=kmeans.labels_.astype(float), s=25, alpha=0.3)
plt.scatter(centroids[:, 0], centroids[:, 1], c=['r'], s=200, edgecolors='black')
plt.title('K-Means on Education Years vs. Age (2 Clusters)')
plt.xlabel('Normalized Age (mean: 38.6 yrs)')
plt.ylabel('Normalized Education Years (mean: 10.1 yrs)')
plt.show()
print('')

#######################################
#using the split_dataset Auxilary function as defined in L08-PredictiveModels lab
#to make sure my testing and training data are not the same
#modified to work with dataframes
def split_dataset(data, r): # split a dataset in matrix format, using a given ratio for the testing set
	N = len(data)	
	X = []
	Y = []
	
	if r >= 1: 
		print ("Parameter r needs to be smaller than 1!")
		return
	elif r <= 0:
		print ("Parameter r needs to be larger than 0!")
		return

	n = int(round(N*r)) # number of elements in testing sample
	nt = N - n # number of elements in training sample
	ind = -np.ones(n,int) # indexes for testing sample
	R = np.random.randint(N) # some random index from the whole dataset
	
	for i in range(n):
		while R in ind: R = np.random.randint(N) # ensure that the random index hasn't been used before
		ind[i] = R

	ind_ = list(set(range(N)).difference(ind)) # remaining indexes	
	X = data.iloc[ind_,:-1] # training features
	XX = data.iloc[ind,:-1] # testing features
	Y = data.iloc[ind_,-1] # training targets
	YY = data.iloc[ind,-1] # testing targets
	return X, XX, Y, YY

#######################################
"""Nearest Neighbors regression model"""
#Applying Nearest Neighbors regression model
print ('\n\n\nNearest Neighbors Regression\n')

r = 0.15 # ratio of test data over all data (this can be changed to any number between 0.0 and 1.0 (not inclusive)
X, XX, Y, YY = split_dataset(Census, r)
X = X.values
Y = Y.values

k = 10 # number of neighbors to be used
distance_metric = 'euclidean'
NN = KNeighborsRegressor(n_neighbors=k, metric=distance_metric)
NN.fit(X, Y)
predictions = NN.predict(XX)
target_values = np.array(YY)
print ("predictions for Regression test set:")
print (predictions)
print ('actual target values:')
print (target_values)

#######################################
"""Logistic Regression classifier model"""
#Applying Logistic Regression classifier model
print ('\n\n\nLogistic Regression Classifier\n')

r = 0.2 # ratio of test data over all data (this can be changed to any number between 0.0 and 1.0 (not inclusive)
X, XX, Y, YY = split_dataset(Census, r)
X = X.values
Y = Y.values

clf = DecisionTreeClassifier() # default parameters are fine
clf.fit(X, Y) 
predictions = clf.predict(XX)
target_values = np.array(YY)
print ("predictions for Classification test set:")
print (predictions)
print ('actual target values:')
print (target_values)

#accuracy rate:
accuracy = accuracy_score(target_values,predictions)
print("\n\nThe accuracy rate for the classifier model is:")
print(accuracy)

#######################################
"""Confusion Matrix"""
CM = confusion_matrix(target_values,predictions)
print ("\n\nConfusion matrix:\n", CM)
tn, fp, fn, tp = CM.ravel()
print ("\nTP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(target_values, predictions)
print ("\nAccuracy rate:", AR)
ER = 1.0 - AR
print ("\nError rate:", ER)
P = precision_score(target_values, predictions)
print ("\nPrecision:", np.round(P, 2))
R = recall_score(target_values, predictions)
print ("\nRecall:", np.round(R, 2))
F1 = f1_score(target_values, predictions)
print ("\nF1 score:", np.round(F1, 2))

#######################################
"""ROC Curve"""
#using example code from the L09-AccuracyMeasures lab and modifying it to 
#use my variables
LW = 1.5 # line width for plots
LL = "lower right" # legend location
LC = 'darkgreen' # Line Color

fpr, tpr, th = roc_curve(target_values, predictions) # False Positive Rate, True Posisive Rate, probability thresholds
AUC = auc(fpr, tpr)
print ("\nTP rates:", np.round(tpr, 2))
print ("\nFP rates:", np.round(fpr, 2))
print ("\nProbability thresholds:", np.round(th, 2))

plt.figure()
plt.title('ROC curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE Positive Rate')
plt.ylabel('TRUE Positive Rate')
plt.plot(fpr, tpr, color=LC,lw=LW, label='ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--') # reference line for random classifier
plt.legend(loc=LL)
plt.show()

#The ROC curve plotted shows that my model is OK at predicting the true values
#but there are still more false positives relative to true positives

print ("\nAUC score (using auc function):", np.round(AUC, 2))
print ("\nAUC score (using roc_auc_score function):", np.round(roc_auc_score(target_values, predictions), 2), "\n")

#Final conclusions:

#I could definitely improve the accuracy of my model. The false positives overcome
#the true positives, shown by the ROC curve, and ideally this trend would be reversed.
#I would be able to use my model with the assumption that the results are not 
#to be taken too seriously.