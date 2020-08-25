"""
Natalie Moore
Assignment L07

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

BINARY QUESTION: Do most US adults make more than $50k/year?

ANSWER: No, there are many more data points for salaries less than or equal to 
$50k/year. Also, the cluster centroids are centered in the lower half of the plot,
signifying that most of the data points exist there (in the lower salary range).

NON-BINARY QUESTION: How do age and hours worked/week correlate, if any?

ANSWER: Generally, it looks like people of all ages work around 40/hrs a week 
but younger people (than 38 yrs) work slightly less than that since one of the 
cluster centroids is below the mean hours/week. There is also hardly any oldest
or youngest people who work more than 70 hrs/week. The general correlation looks
like an offset parabola. 


Below I will perform a K-means grouping analysis with the sklearn package for 
the attributes relating to my questions above: Age, Hours/Week, and Salary.

"""

import pandas as pd #importing necessary packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans as Kmeans

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

#creating variable names for integer columns and 
#normalizing them using Z normalization
Age = Census.loc[:,"Age"]
Age_norm = (Age - np.mean(Age))/np.std(Age)
Education_num = Census.loc[:,"Education(num)"]
Education_num_norm = (Education_num - np.mean(Education_num))/np.std(Education_num)
Census.rename(columns={'Hours/Week': 'Hours'}, inplace=True)
Hours = Census.loc[:,"Hours"]
Hours_norm = (Hours - np.mean(Hours))/np.std(Hours)

#creating variable names for object columns
Class = Census.loc[:,"WorkClass"]
Education = Census.loc[:,"Education"]
Marital = Census.loc[:,"MaritalStatus"]
Occupation = Census.loc[:,"Occupation"]
Relationship = Census.loc[:,"Relationship"]
Race = Census.loc[:,"Race"]
Sex = Census.loc[:,"Sex"]
Native = Census.loc[:,"NativeCountry"]
Salary = Census.loc[:,"Salary"]

#COMMENT OUT IF YOU WANTED TO USE RACE ATTRIBUTE
##de-coding the Race column to use for Kmeans
#Census.loc[Race == " White", "Race"] = 0
#Census.loc[Race == " Black", "Race"] = 1
#Census.loc[Race == " Asian-Pac-Islander", "Race"] = 2
#Census.loc[Race == " Amer-Indian-Eskimo", "Race"] = 3
#Census.loc[Race == " Other", "Race"] = 4
##renaming Race column to be Race_decoded
#Census.rename(columns={'Race': 'Race_decoded'}, inplace=True)
#Race_decoded = Census.loc[:,"Race_decoded"]
#Race_decoded = (Race_decoded - np.mean(Race_decoded))/np.std(Race_decoded)

#de-coding the Salary column to use for Kmeans
#no normalization necessary since it is a categorical variable
Census.loc[Salary == " <=50K","Salary"] = 0
Census.loc[Salary == " >50K","Salary"] = 1
Census.rename(columns={'Salary': 'Salary_decoded'}, inplace=True)
Salary_decoded = Census.loc[:,"Salary_decoded"]

#creating a newdataframe of Age (normalized) and Salary (decoded) to use for Kmeans
Variables2Compare_1 = pd.concat([Age_norm, Salary_decoded], axis=1)

#using the Kmeans package to fit the data to 2 data clusters
#using the Kmeans package to find the cluster coordinates
kmeans = Kmeans(n_clusters=2).fit(Variables2Compare_1)
centroids_1 = kmeans.cluster_centers_
print('Centroid coordinates Figure 1:')
print(centroids_1)

#Figure parameters to make them bigger
plt.rcParams["figure.figsize"] = [10, 10]
matplotlib.rcParams.update({'font.size': 20})

#plotting the normalized Age and decoded Salary attributes against each other
#the colors of the points are determined by the Kmeans package (meaning each point is assigned a label from Kmeans)
plt.scatter(Variables2Compare_1['Age'],Variables2Compare_1['Salary_decoded'], c=kmeans.labels_.astype(float), s=25, alpha=0.3)
plt.scatter(centroids_1[:, 0], centroids_1[:, 1], c=['r'], s=200, edgecolors='black')
plt.title('Figure 1: Kmeans on Salary vs. Age (2 Clusters)')
plt.xlabel('Normalized Age (mean: 38.6 yrs)')
plt.ylabel('Salary (0:<=50K, 1:>50K)')
plt.show()
print('')

#creating a new dataframe of Age (normalized) and Hours/Week (normalized) to use for Kmeans
Variables2Compare_2 = pd.concat([Hours_norm, Age_norm], axis=1)
  
#using the Kmeans package to fit the data to 2 data clusters
#using the Kmeans package to find the cluster coordinates
kmeans = Kmeans(n_clusters=2).fit(Variables2Compare_2)
centroids_2 = kmeans.cluster_centers_
print('Centroid coordinates Figure 2:')
print(centroids_2)

#plotting the normalized Age and Salary attributes against each other
#the colors of the points are determined by the Kmeans package (meaning each point is assigned a label from Kmeans)
plt.scatter(Variables2Compare_2['Age'],Variables2Compare_2['Hours'], c=kmeans.labels_.astype(float), s=25, alpha=0.3)
plt.scatter(centroids_2[:, 0], centroids_2[:, 1], c=['r'], s=200, edgecolors='black')
plt.title('Figure 2: Kmeans on Hours Worked/Week vs. Age (2 Clusters)')
plt.ylabel('Normalized Hours/Week (mean: 40.4 hrs/week)')
plt.xlabel('Normalized Age (mean: 38.6 yrs)')
plt.show()
