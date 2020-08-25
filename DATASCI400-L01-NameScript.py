" #Natalie Moore
" #L01

#Running this script will print a statement with my name and the current date and time.


import datetime as dt #so I can use the datetime package to get the current date and time

def my_name(): #creating a function called my_name that will print my name 
    print ("My name is Natalie Moore.") #printing my name

def date_and_time(): #creating a function called date_and_time that will print the current date and time
    now = str(dt.datetime.now()) #creating a variable called now that uses the datetime package and turns it into a string
    print ("The current date and time is " + now) #printing the date and time in GMT

my_name() #calling the function my_name
date_and_time() #calling the function date_and_time
