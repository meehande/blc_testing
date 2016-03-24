# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 09:53:23 2016

@author: Deirdre Meehan
"""
"""
PARSER FOR MOVIELENS DATA IN FORM:
USER_ID::MOVIE_ID::RATING::TIMESTAMP
"""
import numpy as np

path = '../ml-10M100K/'
filename = 'ra.test'
delimiter = '::' 
#def readR(path, filename, delimiter):    
f = open(path+filename, 'r')

#movie_ids = [float(line.split('::')[1]) for line in f]
#m = max(movie_ids)
#find the max movie id 
#find the #users
#initialise R with zeros and go from there!
movie_ids = []
user_ids = []
for line in f:#R: each user is a row, each column is an item, each value is a rating 
    values = line.split(delimiter)
    movie_ids.append(int(values[1]))
    user_ids.append(int(values[0]))
    
m = max(movie_ids)
n = max(user_ids)
print "n, m: ", n, "  ", m
R = np.zeros((n+1,m+1))
#float16 is way smaller - float 64 R was 5,896,880; float16 was 1,474,304 (bytes)

f.seek(0)#go back to start of file

for line in f:
    values = line.split(delimiter)
    R[values[0],values[1]] = float(values[2]) # userid, movieid = rating
#return R