#@author: natal
# -*- coding: utf-8 -*-
#Mon May 20 19:30:46 2019

"""
Natalie Moore
L05 Assignment

This dataset is from the UCI ML archive. 
The data is from a 1996 US Census.

Unfortunately, I did not have to decode any variables. I could have easily by
assigning the coded values to a category, just like I did below with missing
values and consolidation. 

By using variable.unique() I was able to find three variables that contained
missing values: WorkClass, Occupation, and NativeCountry. I decided the best
way to replace these missing values was to impute them onto the most frequent
value of the variable, found by using variable.value_counts().

I also decided to consolidate the NativeCountry column, since it contained so
many countries. I decided to change this column to NativeContinent instead and
consolidate each country into categories based on which continents they are in.

For the one-hot encoding, I decided to create dummy variables for each value
of the Race column. In doing so I was able to create new columns for each Race
value that are binary and therefore turn the previous categorical Race column
into integer columns. No data was lost.

The plotting at the end shows interesting trends in the Census:
    NativeContinent was plotted and showed that most people filling out the 
    census were from North America.
    MaritalStatus was plotted and showed that most people filling out the 
    census were married.
    Sex was plotted and showed a male-trend in the census data.
"""

import pandas as pd #importing necessary packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

#creating my dataframe called Census from the UCI archive
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
Census = pd.read_csv(url, header=None) 
Census.columns = ["Age","WorkClass","FinalWeight","Education",
                  "Education(num)","MaritalStatus","Occupation",
                  "Relationship","Race","Sex","Capital-Gain",
                  "Capital-Loss","Hours/Week","NativeCountry","Salary"]
Census = Census.drop("FinalWeight", axis = 1) #dropping FinalWeight column because it's not relavent/useful 

#creating variable names for integer columns and 
#normalizing them using Z normalization
Age = Census.loc[:,"Age"]
Age = (Age - np.mean(Age))/np.std(Age)
Education_num = Census.loc[:,"Education(num)"]
Education_num = (Education_num - np.mean(Education_num))/np.std(Education_num)
CapitalGain = Census.loc[:,"Capital-Gain"]
CapitalGain = (CapitalGain - np.mean(CapitalGain))/np.std(CapitalGain)
CapitalLoss = Census.loc[:,"Capital-Loss"]
CapitalLoss = (CapitalLoss - np.mean(CapitalLoss))/np.std(CapitalLoss)
Hours = Census.loc[:,"Hours/Week"]
Hours = (Hours - np.mean(Hours))/np.std(Hours)

#binning the Age column (with 6 bins) from the Census dataframe
freq, bounds = np.histogram(Census.loc[:,"Age"], 6)
#showing the results in a histogram:
plt.hist(Census.loc[:,"Age"], bounds, label = 'Age', color='c')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of Census Age with 6 Bins')
plt.legend()
plt.rcParams["figure.figsize"] = [12,8]
plt.show()

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

#showing the value counts of the categorical variables that contain missing values
#describing that I will impute missing values onto the most frequent value of the variable
print("WorkClass value counts show I should impute the missing values onto 'Private':")
print("")
print(Class.value_counts())
print("")
print("Occupation value counts show I should impute the missing values onto 'Prof-specialty':")
print("")
print(Occupation.value_counts())
print("")
print("NativeCountry value counts show I should impute the missing values onto 'United-States':")
print("")
print(Native.value_counts())

#imputing missing values onto most frequent value
Census.loc[Class == " ?", "WorkClass"] = " Private"
Census.loc[Occupation == " ?", "Occupation"] = " Prof-specialty"
Census.loc[Native == " ?", "NativeCountry"] = " United-States"

#consolidating the NativeCountry column into NativeContinents
Census.loc[Native == " United-States", "NativeCountry"] = "North-America"
Census.loc[Native == " Mexico", "NativeCountry"] = "North-America"
Census.loc[Native == " Canada", "NativeCountry"] = "North-America"
Census.loc[Native == " El-Salvador", "NativeCountry"] = "North-America"
Census.loc[Native == " Nicaragua", "NativeCountry"] = "North-America"
Census.loc[Native == " Cuba", "NativeCountry"] = "North-America"
Census.loc[Native == " Jamaica", "NativeCountry"] = "North-America"
Census.loc[Native == " Haiti", "NativeCountry"] = "North-America"
Census.loc[Native == " Trinadad&Tobago", "NativeCountry"] = "North-America"
Census.loc[Native == " Honduras", "NativeCountry"] = "North-America"
Census.loc[Native == " Puerto-Rico", "NativeCountry"] = "North-America"
Census.loc[Native == " Guatemala", "NativeCountry"] = "North-America"
Census.loc[Native == " Dominican-Republic", "NativeCountry"] = "North-America"

Census.loc[Native == " Ecuador", "NativeCountry"] = "South-America"
Census.loc[Native == " Peru", "NativeCountry"] = "South-America"
Census.loc[Native == " Columbia", "NativeCountry"] = "South-America"

Census.loc[Native == " Thailand", "NativeCountry"] = "Asia"
Census.loc[Native == " Hong", "NativeCountry"] = "Asia"
Census.loc[Native == " Iran", "NativeCountry"] = "Asia"
Census.loc[Native == " Taiwan", "NativeCountry"] = "Asia"
Census.loc[Native == " Japan", "NativeCountry"] = "Asia"
Census.loc[Native == " Vietnam", "NativeCountry"] = "Asia"
Census.loc[Native == " China", "NativeCountry"] = "Asia"
Census.loc[Native == " India", "NativeCountry"] = "Asia"
Census.loc[Native == " Philippines", "NativeCountry"] = "Asia"
Census.loc[Native == " Cambodia", "NativeCountry"] = "Asia"
Census.loc[Native == " Laos", "NativeCountry"] = "Asia"
Census.loc[Native == " Cambodia", "NativeCountry"] = "Asia"

Census.loc[Native == " Germany", "NativeCountry"] = "Europe"
Census.loc[Native == " England", "NativeCountry"] = "Europe"
Census.loc[Native == " Italy", "NativeCountry"] = "Europe"
Census.loc[Native == " Poland", "NativeCountry"] = "Europe"
Census.loc[Native == " France", "NativeCountry"] = "Europe"
Census.loc[Native == " Greece", "NativeCountry"] = "Europe"
Census.loc[Native == " Ireland", "NativeCountry"] = "Europe"
Census.loc[Native == " Scotland", "NativeCountry"] = "Europe"
Census.loc[Native == " Holand-Netherlands", "NativeCountry"] = "Europe"
Census.loc[Native == " Hungary", "NativeCountry"] = "Europe"
Census.loc[Native == " Portugal", "NativeCountry"] = "Europe"
Census.loc[Native == " Yugoslavia", "NativeCountry"] = "Europe"

Census.loc[Native == " South", "NativeCountry"] = "Other"
Census.loc[Native == " Outlying-US(Guam-USVI-etc)", "NativeCountry"] = "Other"

#renaming NativeCountry column to reflect consolidation
Census.rename(columns={'NativeCountry': 'NativeContinent'}, inplace=True)
Native = Census.loc[:,"NativeContinent"]

#one-hot encoding the Race column
White = Census.loc[:, "White"] = (Race == " White").astype(int)
Black = Census.loc[:, "Black"] = (Race == " Black").astype(int)
AsianPacIslander = Census.loc[:, "Asian-Pac-Islander"] = (Race == " Asian-Pac-Islander").astype(int)
AmerIndEsk = Census.loc[:, "Amer-Ind-Esk"] = (Race == " Amer-Indian-Eskimo").astype(int)
Other = Census.loc[:, "Other"] = (Race == " Other").astype(int)

#dropping the race column since its now obsolete
Census = Census.drop("Race", axis = 1)

#plotting some of the categorical columns
Native.value_counts().plot(kind = 'barh') #makes sense since this is a US Census
print("")
Sex.value_counts().plot(kind = 'barh') #many more men than women answered the Census
print("")
Marital.value_counts().plot(kind = 'barh') #most people who answered Census are married