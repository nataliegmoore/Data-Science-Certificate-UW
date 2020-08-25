""" 
#Natalie Moore
#L03

#This code creats two random arrays, arr1 and arr2. 
#arr1 is an int array that contains outliers.
#arr2 is a string array that contains improper missing values.
#The functions below remove and replace the outliers in arr1 with the mean value of non-outliers,
#as well as replacing the improper values of arr2 with the median of arr2's values.
#I had to convert arr2 from a string array into an int array to get the median.

#This code prints the arrays and their sizes first.
#Then, it prints the replacement arrays and their sizes for comparison. 
"""

import numpy as np #importing necessary package Numpy

#creating random numpy array called arr1 with integer values between 1 and 10, size 40.
arr1 = np.random.randint(1, 10, 40) 

arr1[10] = 666 #making the 11th element in arr1 an outlier
arr1[15] = 777 #making the 16th element in arr1 an outlier
arr1[20] = 888 #making the 21st element in arr1 an outlier
arr1[35] = 999 #making the 36th element in arr1 an outlier


#creating random numpy array called arr2 with string values between 1 and 10, size 40.
#note: I made arr2 a string so I could make certain elements improper missing values
arr2 = (np.random.randint(1, 10, 40)).astype(str)

arr2[10] = "" #making the 11th element in arr2 an improper missing value
arr2[15] = "?" #making the 16th element in arr2 an improper missing value
arr2[20] = " " #making the 21st element in arr2 an improper missing value
arr2[35] = "nan" #making the 36th element in arr2 an improper missing value

print ("arr1 = ") #print statements for clarity
print (arr1.astype(str))
size1 = str(np.size(arr1))
print ("The size of arr1 with outliers is " + size1 + ".")

print ("")

print ("arr2 = ") #print statements for clarity
print (arr2)
size2 = str(np.size(arr2))
print ("The size of arr2 with improper missing values is " + size2 + ".") #print statements for clarity

print ("")
print ("------------C-L-E-A-N-I-N-G---D-A-T-A-----------------------")

LimitHi = np.mean(arr1) + 2*np.std(arr1) #Upper limit for arr1 values
LimitLo = np.mean(arr1) - 2*np.std(arr1) #Lower limit for arr1 values
FlagGood1 = (arr1 >= LimitLo) & (arr1 <= LimitHi) #Defining the flags for arr1 values that are within limits
FlagBad1 = ~FlagGood1 #Defining the flags for arr1 values that are outside limits

FlagGood2 = [element.isdigit() for element in arr2] #Defining the flags for arr2 values that are integer elements
FlagBad2 = [not element.isdigit() for element in arr2] #Defining the flags for arr2 values that are not int elements

def remove_outlier(arr1): #Defining function called remove_outlier for arr1 that removes the outliers above
    arr1 = arr1[FlagGood1]  #Indexing arr1 through the Boolean array called FlagGood to select only the good flags
    print ("arr1[Removed] = ")
    print (arr1.astype(str))
    size1 = str(np.size(arr1))
    print ("The size of arr1 after removing outliers is " + size1 + ".")

def replace_outlier(arr1): #Defining function called replace_outlier for arr1 that replaces the outliers above
    arr1[FlagBad1] = np.mean(arr1[FlagGood1]) #Indexing the arr1 bad flags through the good flags' mean value to replace them with said mean
    print ("arr1[Replaced] = ")
    print (arr1.astype(str))
    size1 = str(np.size(arr1))
    print ("The size of arr1 after replacing outliers is " + size1 + ".")
    
def fill_median(arr2): #Defining function called fill_median for arr2 that replaces missing values with the median of the good values
    arr2[FlagBad2] = np.median(arr2[FlagGood2].astype(int)) #Indexing the arr2 bad flags through the good flags' median value (as ints) and replacing them with sain median
    print ("arr2[Replaced] = ")
    print (arr2.astype(str))
    size2 = str(np.size(arr2))
    print ("The size of arr2 after replacing outliers is " + size2 + ".")
    

print ("")
remove_outlier(arr1) #Calling the remove_outlier function 
print ("")
replace_outlier(arr1) #Calling the replace_outlier function
print ("")
fill_median(arr2) #Calling the fill_median function
