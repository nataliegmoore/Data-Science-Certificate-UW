"""    
Natalie Moore
Assignment L08

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
removing outliers. 

BINARY QUESTION: By using the Logistic Regression classification model, can I 
accurately (greater than 80%) predict whether people of non-white races make 
$50K/yr or more?

ANSWER: No, the accuracy rate (printed last) is always around 75%. 

    
Below I will first perform K-Means on the Age and Education_num attributes to 
see the relationship between a US adult's age and how many years of education
they've completed.

After K-Means is applied, I will then use the Nearest Neighbors regression 
model to train and test the Census data and predict the hours/week worked by
US adults. I will also use the Logistic Regression classification model to 
train and test the Census data and predict the salary of adults with different
racial backgrounds.
"""
#importing necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans as Kmeans
from sklearn.metrics import accuracy_score
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

#Census_class is used for the classification model
#Census_regress is used for the regression model
Census_class = pd.concat([Census.loc[:,"Race"], Census.loc[:,"Salary"]], axis = 1)
Census_regress = pd.concat([Census.loc[:,"Age"], Census.loc[:,"Education(num)"], Census.loc[:,"Hours/Week"]], axis = 1)

#de-coding the Race column to use for Logistic Regression Classifier
Census_class.loc[Census_class.loc[:,"Race"] == " White", "Race"] = 0
Census_class.loc[Census_class.loc[:,"Race"] == " Black", "Race"] = 1
Census_class.loc[Census_class.loc[:,"Race"] == " Asian-Pac-Islander", "Race"] = 2
Census_class.loc[Census_class.loc[:,"Race"] == " Amer-Indian-Eskimo", "Race"] = 3
Census_class.loc[Census_class.loc[:,"Race"] == " Other", "Race"] = 4
#renaming Race column to be Race_decoded
Census_class.rename(columns={'Race': 'Race_decoded'}, inplace=True)

#de-coding the Salary column to use for Logistic Regression Classifier
Census_class.loc[Census_class.loc[:,"Salary"] == " <=50K", "Salary"] = 0
Census_class.loc[Census_class.loc[:,"Salary"] == " >50K", "Salary"] = 1
#renaming Salary column to be Salary_decoded
Census_class.rename(columns={'Salary': 'Salary_decoded'}, inplace=True)


#using normalization function from Labs but modified for a dataframe
def normalize(X): # max-min normalizing
    N, m = X.shape
    Y = np.zeros([N, m])
    
    for i in range(m):
        mX = min(X.iloc[:,i])
        Y[:,i] = (X.iloc[:,i] - mX) / (max(X.iloc[:,i]) - mX)
    
    return Y

#normalized dataset as a dataframe
Census_regress = pd.DataFrame(normalize(Census_regress.loc[:,:]))
Census_regress.rename(columns={0: 'Age_norm'}, inplace=True)
Census_regress.rename(columns={1: 'Education_norm'}, inplace=True)
Census_regress.rename(columns={2: 'Hours_norm'}, inplace=True)
#creating variable names for integer columns
Age_norm = Census_regress.loc[:,'Age_norm']
Education_norm = Census_regress.loc[:,'Education_norm']
Hours_norm = Census_regress.loc[:,'Hours_norm']

#######################################
"""K-Means"""
#creating a newdataframe of Age (normalized) and Education_num (normalized) 
#to use for Kmeans
Variables4Kmeans = pd.concat([Age_norm, Education_norm], axis = 1)

#using the Kmeans package to fit the data to 2 data clusters
#using the Kmeans package to find the cluster coordinates
kmeans = Kmeans(n_clusters=2).fit(Variables4Kmeans)
centroids = kmeans.cluster_centers_
print('Centroid coordinates Figure 1:')
print(centroids)

#plotting the normalized Age and Education_num attributes against each other
#the colors of the points are determined by the Kmeans package (meaning each point is assigned a label from Kmeans)
plt.scatter(Variables4Kmeans['Age_norm'],Variables4Kmeans['Education_norm'], c=kmeans.labels_.astype(float), s=25, alpha=0.3)
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
X, XX, Y, YY = split_dataset(Census_regress, r)
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

r = 0.15 # ratio of test data over all data (this can be changed to any number between 0.0 and 1.0 (not inclusive)
X, XX, Y, YY = split_dataset(Census_class, r)
X = X.values
Y = Y.values

C_parameter = 50. / len(X) # parameter for regularization of the model
class_parameter = 'ovr' # parameter for dealing with multiple classes
penalty_parameter = 'l1' # parameter for the optimizer (solver) in the function
solver_parameter = 'saga' # optimization system used
tolerance_parameter = 0.1 # termination parameter
clf = LogisticRegression(C=C_parameter, multi_class=class_parameter, penalty=penalty_parameter, solver=solver_parameter, tol=tolerance_parameter)
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