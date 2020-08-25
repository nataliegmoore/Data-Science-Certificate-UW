"""
Natalie Moore
L04

This code uses data from the University of California at Irvine's machine learning database.
My chosen dataset contains data on breast cancer of patients in Wisconsin, therefore my
dataset was named BreastCancer.

Unfortunately, none of the attributes contained outliers. This is confirmed by the standard deviation
of each attribute (calculated below). If there were outliers, I would've set the highest value of the
related attribute to not include the outlier. The outlier would be evident in the histogram:

TooHigh = BreastCancer.loc[:, 'attribute'] > histogram value
BreastCancer.loc[TooHigh, 'attribute'] = histogram value

The attribute which required imputation of missing values was the Bare Nuclei attribute. I knew this 
because the data type of that attribute was "object" instead of a numeric type. Furthermore, when I
showed the unique values of said attribute I could see there was a ? instead of a number.

I plotted histograms of every attribute because they were all numeric types. However, I could have 
just plotted all of them but Sample and Class. Sample is a key attribute, so each value only occurs 
once. Class is a binary attribute and a histogram was not very useful since there were only two values.

I decided not to remove any attributes because they all contain valuable information about the breast
cancer cells. If I wanted I could have removed the Bare Nuclei attribute, which would have removed 
all the non-numeric values at once.

I also did not remove any rows because there was only one attribute with non-numeric values. If I 
removed any rows, I would only be removing 1 non-numeric value per row removed.
"""

#importing relavent packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

#importing the data from the ics web database and defining the column names
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
BreastCancer = pd.read_csv(url, header=None)
BreastCancer.columns = ["Sample", "Clump Thickness", "Unif Cell Size", "Unif Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]

#print statements
print ("Here are the column names and types of my dataset called BreastCancer:")
print ("")
print (BreastCancer.dtypes) #showing the names and columns of my dataset
print ("")
print ("The column named Bare Nuclei should be a numeric type. There must be missing values present.", "Below lists the unique values of that column and the sum of its missing values:")
print ("")
print (BreastCancer.loc[:,'Bare Nuclei'].unique()) #showing the unique values in the Bare Nuclei column
print (sum(BreastCancer.loc[:,'Bare Nuclei'] == "?")) #summing up all the non-numeric values in the Bare Nuclei column

BreastCancer.loc[:, 'Bare Nuclei'] = pd.to_numeric(BreastCancer.loc[:, 'Bare Nuclei'], errors='coerce') #changing all non-numeric values from the column Bare Nuclei to NaN values
HasNan = np.isnan(BreastCancer.loc[:, 'Bare Nuclei']) #defining where the NaN values are in the Bare Nuclei column
BreastCancer.loc[HasNan, 'Bare Nuclei'] = np.nanmedian(BreastCancer.loc[:,'Bare Nuclei']) #replacing the NaN values with the median value calculated from the numeric values in the Bare Nuclei column

print ("")
print ("After assigning median values for missing numeric values, here are the column names and types now:")
print ("")
print (BreastCancer.dtypes) #showing the names and columns of my dataset after non-numeric values have been replaced by the median

#histogram plots
print("")
print("Below are the histograms for each column in order.")
print("")
plt.hist(BreastCancer.loc[:, 'Sample']) #Each value in the Sample column occurs once, so the histogram only has one bin. 
plt.title('Sample')
plt.show()
plt.hist(BreastCancer.loc[:, 'Clump Thickness']) 
plt.title('Clump Thickness')
plt.show()
plt.hist(BreastCancer.loc[:, 'Unif Cell Size']) 
plt.title('Unif Cell Size')
plt.show()
plt.hist(BreastCancer.loc[:, 'Unif Cell Shape'])
plt.title('Unif Cell Shape')
plt.show()
plt.hist(BreastCancer.loc[:, 'Marginal Adhesion'])
plt.title('Marginal Adhesion')
plt.show()
plt.hist(BreastCancer.loc[:, 'Single Epithelial Cell Size'])
plt.title('Single Epithelial Cell Size')
plt.show()
plt.hist(BreastCancer.loc[:, 'Bare Nuclei'])
plt.title('Bare Nuclei')
plt.show()
plt.hist(BreastCancer.loc[:, 'Bland Chromatin'])
plt.title('Bland Chromatin')
plt.show()
plt.hist(BreastCancer.loc[:, 'Normal Nucleoli'])
plt.title('Normal Nucleoli')
plt.show()
plt.hist(BreastCancer.loc[:, 'Mitoses'])
plt.title('Mitoses')
plt.show()
plt.hist(BreastCancer.loc[:, 'Class']) #Class is a binary type. 
plt.title('Class')
plt.show()

print ("")
print("After reviewing the distributions of each variable, I see no outliers to replace.")

#scatter plot
print("")
print("Here is a scatter plot of Clump Thickness vs. Uniform Cell Size:")
plt.scatter(BreastCancer.loc[:, 'Clump Thickness'], BreastCancer.loc[:, 'Unif Cell Size'])
plt.xlabel('Clump Thickness')
plt.ylabel('Unif Cell Size')
plt.show()

#standard deviations of each variable and printing them
print ("")
print ("Here are the standard deviations of each numeric variable, excluding Sample and Class:")
print ("")

clumpthickness = str(np.std(BreastCancer.loc[:, 'Clump Thickness']))
print ("Clump Thickness: " + clumpthickness)

cellsize = str(np.std(BreastCancer.loc[:, 'Unif Cell Size']))
print ("Unif Cell Size: " + cellsize)

cellshape = str(np.std(BreastCancer.loc[:, 'Unif Cell Shape']))
print ("Unif Cell Shape: " + cellshape)

ma = str(np.std(BreastCancer.loc[:, 'Marginal Adhesion']))
print ("Marginal Adhesion: " + ma)

secz = str(np.std(BreastCancer.loc[:, 'Single Epithelial Cell Size']))
print ("Single Epithelial Cell Size: " + secz)

barenuclei = str(np.std(BreastCancer.loc[:, 'Bare Nuclei']))
print ("Bare Nuclei: " + barenuclei)

bland = str(np.std(BreastCancer.loc[:, 'Bland Chromatin']))
print ("Bland Chromatin: " + bland)

normalnuc = str(np.std(BreastCancer.loc[:, 'Normal Nucleoli']))
print ("Normal Nucleoli: " + normalnuc)

mitoses = str(np.std(BreastCancer.loc[:, 'Mitoses']))
print ("Mitoses: " + mitoses)
