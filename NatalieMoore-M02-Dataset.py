"""
Natalie Moore
Milestone #2
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
#below are removed b/c outlier values revealed the only non-outliers == 0
Census = Census.drop("Capital-Gain", axis = 1) #dropping Capital-Gain column because it's mostly 0
Census = Census.drop("Capital-Loss", axis = 1) #dropping Capital-Loss column because it's mostly 0

#creating variable names for integer columns and 
#normalizing them using Z normalization
Age = Census.loc[:,"Age"]
Age = (Age - np.mean(Age))/np.std(Age)
Education_num = Census.loc[:,"Education(num)"]
Education_num = (Education_num - np.mean(Education_num))/np.std(Education_num)
Hours = Census.loc[:,"Hours/Week"]
Hours = (Hours - np.mean(Hours))/np.std(Hours)

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

#only the categorical vars need to be imputed
#showing the value counts of the categorical variables that contain missing values
#describing that I will impute missing values onto the most frequent value of the variable
#print("WorkClass value counts show I should impute the missing values onto 'Private':")
#print("")
#print(Class.value_counts())
#print("")
#print("Occupation value counts show I should impute the missing values onto 'Prof-specialty':")
#print("")
#print(Occupation.value_counts())
#print("")
#print("NativeCountry value counts show I should impute the missing values onto 'United-States':")
#print("")
#print(Native.value_counts())

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

#consolidating Education column (binning the column)
Census.loc[Education == " Preschool", "Education"] = "Pre-HighSchool"
Census.loc[Education == " 1st-4th", "Education"] = "Pre-HighSchool"
Census.loc[Education == " 5th-6th", "Education"] = "Pre-HighSchool"
Census.loc[Education == " 7th-8th", "Education"] = "Pre-HighSchool"
Census.loc[Education == " 9th", "Education"] = "HighSchool"
Census.loc[Education == " 10th", "Education"] = "HighSchool"
Census.loc[Education == " 11th", "Education"] = "HighSchool"
Census.loc[Education == " 12th", "Education"] = "HighSchool"
Census.loc[Education == " HS-grad", "Education"] = "HighSchool"
Census.loc[Education == " Some-college", "Education"] = "AnyCollege"
Census.loc[Education == " Bachelors", "Education"] = "AnyCollege"
Census.loc[Education == " Assoc-voc", "Education"] = "AnyCollege"
Census.loc[Education == " Assoc-acdm", "Education"] = "AnyCollege"
Census.loc[Education == " Prof-school", "Education"] = "AnyCollege"
Census.loc[Education == " Masters", "Education"] = "Post-College"
Census.loc[Education == " Doctorate", "Education"] = "Post-College"

Census.rename(columns={'Education': 'GeneralEducation'}, inplace=True)
GeneralEducation = Census.loc[:,"GeneralEducation"]

#de-coding the Race column to use for Kmeans
Census.loc[Race == " White", "Race"] = 0
Census.loc[Race == " Black", "Race"] = 1
Census.loc[Race == " Asian-Pac-Islander", "Race"] = 2
Census.loc[Race == " Amer-Indian-Eskimo", "Race"] = 3
Census.loc[Race == " Other", "Race"] = 4

#renaming Race column to be Race_decoded and normalizing
Census.rename(columns={'Race': 'Race_decoded'}, inplace=True)
Race_decoded = Census.loc[:,"Race_decoded"]
Race_decoded = (Race_decoded - np.mean(Race_decoded))/np.std(Race_decoded)

#Save fixed dataset as NatalieMoore-M02-Dataset.csv
Census.to_csv(r'C:\Users\natal\Desktop\DATA\UW\Assignments\NatalieMoore-M02-Dataset.csv')