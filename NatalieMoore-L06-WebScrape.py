"""
Natalie Moore
Assignment 06: WebScrape


For this assignment I chose a wikipedia URL and used the requests package to 
read the HTML code making up the website and its content. 

I then used the BeautifulSoup package to find all the websites with the 
"external text" tag, which for Wikipedia means https external website links. 

To tally up all my websites, I created an empty dictionary and read in each 
BeautifulSoup element as a string and appended the website link part of the 
element to my dictionary. 
"""

#necessary packages
import requests
from bs4 import BeautifulSoup 

#MyURL is a wiki page for my favorite Star Trek series
MyURL = "https://en.wikipedia.org/wiki/Star_Trek:_Discovery"

print("My URL for this assignment is", MyURL)
print("")

#reading in MyURL to the requests package and scraping the content using
#the requests and BeautifulSoup packages
MyURL = requests.get(MyURL)
content = MyURL.content
soup = BeautifulSoup(content, "lxml")

#using the BeautifulSoup package to find all HTML elements with an <a></a> tag
#and where class="external text"
#which is Wikipedia's class for external web links
all_websites = soup.find_all("a", "external text") 

#creating empty dictionary for my website titles
websites = {}  
#looping through my all_websites result set to read in each element as a string
#and appending that string to my dictionary  
for x in all_websites: 
    title = x.string
    websites[title] = x.attrs['href'] #the href is where the actual website title is

#len(dictionary) finds the length of my dictionary
print("The website tally in my HTML page is", len(websites), "websites.")