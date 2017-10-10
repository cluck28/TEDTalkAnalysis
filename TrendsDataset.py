#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:22:44 2017

@author: Chris
"""

import pandas as pd
import numpy
import pylab as P
import string
import math
import scipy
import random
from sklearn.ensemble import RandomForestClassifier

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('talks.csv', header=0)

#Parse film date to be able ot compare chronologically
date = df['film_date']
monthdict = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
date_out = numpy.zeros(len(date))
for i in range(len(date)):
    date_out[i] = int(date[i].partition('-')[0])+(monthdict[date[i].partition('-')[2]]-1)/12.
     
#Parse upload date to compare chronologically            
date_pub = df['film_date']
date_out_pub = numpy.zeros(len(date_pub))
for i in range(len(date_pub)):
    date_out_pub[i] = int(date_pub[i].partition('-')[0])+(monthdict[date_pub[i].partition('-')[2]]-1)/12.
df['Date Out'] = date_out_pub

#Number of views
views = df['views']
#Fill missing view values with mean value
mean_views = numpy.mean(views)
views.fillna(mean_views, inplace=True) #overwrite views
df['views'].fillna(mean_views, inplace=True) #overwrite dataframe
df['views'].hist()
P.show()

#Mean and median
mean_views = numpy.mean(views)
median_views = numpy.nanmedian(views)
var_views = numpy.std(views)
print mean_views, median_views, var_views

#Any trends versus data posted in views?
P.plot(date_out,views,'g+',date_out_pub,views,'b.')
P.xlabel('Date Published')
P.ylabel('Number of Views')
P.show()

#Parse video themes and look for correlations
tags = df['related_tags']
tags_list = []
for i in range(len(tags)):
    #For each video find the tags
    #Check for missing tags
    if tags[i] == '[]':
        pass
    else:
        s = tags[i].rsplit(',')
        for word in s:
            new_word = word.translate(None, string.punctuation)
            if new_word[0] == ' ':
                hold_list = list(new_word)
                hold_list.pop(0)
                new_word = ''.join(hold_list)
            tags_list.append(''.join(list(new_word)))
            
tags_list = list(set(tags_list)) #tags_list now contains all tags for videos
#Now that I have a look up table of tags loop through and create a new column for each
for k in range(len(tags_list)):
    #Holder for new column to be added
    col = numpy.zeros(len(tags))
    for i in range(len(tags)):
        #Holder for list of tags from this item in dateframe
        tag_item = []
        #Check to see if missing tags
        if tags[i] == '[]':
            pass
        else:
            s = tags[i].rsplit(',')
            for word in s:
                new_word = word.translate(None, string.punctuation)
                if new_word[0] == ' ':
                    hold_list = list(new_word)
                    hold_list.pop(0)
                    new_word = ''.join(hold_list)
                tag_item.append(''.join(list(new_word)))
            for j in range(len(tag_item)):
                if tag_item[j] == tags_list[k]:
                    col[i] = 1.
    df[tags_list[k]] = col
#Special case of an empty list of tags
col = numpy.zeros(len(tags))
for i in range(len(tags)):
    #Check to see if missing tags
    if tags[i] == '[]':
        col[i] = 1.
df['Empty'] = col

#Now have dataframe with columns where 1 means contains that tag and 0 means does not contain that tag
#Just want to add 'Empty' title to tags_list
tags_list.append('Empty')

#Our metric for a successful video is if the number of views are greater than the mean number of views
success = numpy.zeros(len(tags))
for i in range(len(success)):
    if df['views'][i]>mean_views:
        success[i] = 1.
df['Success'] = success

#Let's find the information gain and entropy trying to find a pure group of successful videos
#Starting entropy
success_num = float(len(df[df['Success']==1.]))
fail_num = float(len(df[df['Success']==0.]))
tot_num = success_num+fail_num
print tot_num
success_prob = success_num/tot_num
fail_prob = 1.-success_prob
#Entropy to start
ent_par = -success_prob*math.log(success_prob)-fail_prob*math.log(fail_prob)

#Information gain is calculated as parent entropy - proportion*entropy of children groups

info_list = numpy.zeros(len(tags_list))
#Loop on tags
for i in range(len(tags_list)):
    flag_child1 = 0.
    flag_child2 = 0.
    child1_num = float(len(df[df[tags_list[i]]==1.]))
    #Proportion of total in child1
    child1_prop = child1_num/tot_num
    child1_success = float(len(df[ (df['Success']==1.) & (df[tags_list[i]]==1.) ]))
    child1_fail = child1_num - child1_success
    #Entropy of child1
    #Need to check if child1_success, child1_fail, or child1_num is zero
    if (child1_success != 0.) & (child1_fail != 0.) & (child1_num != 0.):
        ent_child1 = -child1_success/child1_num*math.log(child1_success/child1_num)-child1_fail/child1_num*math.log(child1_fail/child1_num)
    else:
        flag_child1 = 1.
    child2_num = float(len(df[df[tags_list[i]]==0.]))
    #Proportion of total in child2
    child2_prop = child2_num/tot_num
    child2_success = float(len(df[ (df['Success']==1.) & (df[tags_list[i]]==0.) ]))
    child2_fail = child2_num - child2_success
    #Entropy of child2
    if (child2_success != 0.) & (child2_fail != 0.) & (child2_num != 0.):
        ent_child2 = -child2_success/child2_num*math.log(child2_success/child2_num)-child2_fail/child2_num*math.log(child2_fail/child2_num)
    else:
        flag_child2 = 1.
    #Calculate information gain
    if (flag_child1 == 0) & (flag_child2 == 0):
        info_gain = ent_par - (child1_prop*ent_child1 + child2_prop*ent_child2)
    else:
        info_gain = 0
    #print tags_list[i], info_gain
    info_list[i] = info_gain
     
#The maximum information gain             
print tags_list[numpy.argmax(info_list)], info_list[numpy.argmax(info_list)]

#Divide on date
date_cuts = numpy.arange(round(numpy.amin(date_out_pub),0)+1, round(numpy.amax(date_out_pub),0),0.1)
info_list = numpy.zeros(len(date_cuts))
for i in range(len(date_cuts)):
    flag_child1 = 0.
    flag_child2 = 0.
    child1_num = float(len(df[ df['Date Out']>=date_cuts[i] ])) #need a new column in my dataframe! Then select on it
    #Proportion of total in child1
    child1_prop = child1_num/tot_num
    child1_success = float(len(df[ (df['Success']==1.) & (df['Date Out']>=date_cuts[i]) ]))
    child1_fail = child1_num - child1_success
    #Entropy of child1
    #Need to check if child1_success, child1_fail, or child1_num is zero
    if (child1_success != 0.) & (child1_fail != 0.) & (child1_num != 0.):
        ent_child1 = -child1_success/child1_num*math.log(child1_success/child1_num)-child1_fail/child1_num*math.log(child1_fail/child1_num)
    else:
        flag_child1 = 1.
    child2_num = float(len(df[df['Date Out']<date_cuts[i]]))
    #Proportion of total in child2
    child2_prop = child2_num/tot_num
    child2_success = float(len(df[ (df['Success']==1.) & (df['Date Out']<date_cuts[i]) ]))
    child2_fail = child2_num - child2_success
    #Entropy of child2
    if (child2_success != 0.) & (child2_fail != 0.) & (child2_num != 0.):
        ent_child2 = -child2_success/child2_num*math.log(child2_success/child2_num)-child2_fail/child2_num*math.log(child2_fail/child2_num)
    else:
        flag_child2 = 1.
    #Calculate information gain
    if (flag_child1 == 0) & (flag_child2 == 0):
        info_gain = ent_par - (child1_prop*ent_child1 + child2_prop*ent_child2)
    else:
        info_gain = 0
    #print date_cuts[i], info_gain
    info_list[i] = info_gain

#The maximum information gain             
print date_cuts[numpy.argmax(info_list)], info_list[numpy.argmax(info_list)]
#Segment based on date published! Jan 2010
df['Date Good'] = (df['Date Out'] >= 10.0).astype(int)

#Number of comments
comments = df['comments']
def linfunc(x, *params):
    return params[0]+params[1]*x

xvals = numpy.arange(0,numpy.amax(comments))
popt,pcov = scipy.optimize.curve_fit(linfunc, comments, views, p0=(120000,5000))    
P.plot(comments,views,'b.',xvals,linfunc(xvals,*popt),'r')
P.xlabel('Number of Comments')
P.ylabel('Number of Views')
P.show()

#Related videos
titles = df['title']
related = df['related_videos']
score_list = numpy.zeros(len(titles))
#Loop through all rows of dataframe
for j in range(len(titles)):
    #Holder for list of related videos from this item in dateframe
    related_item = []
    #Check to see if missing list of related videos
    if related[j] == '[]':
        pass
    else:
        s = related[j].rsplit(',')
        for word in s:
            new_word = word.translate(None, string.punctuation)
            if new_word[0] == ' ':
                hold_list = list(new_word)
                hold_list.pop(0)
                new_word = ''.join(hold_list)
            related_item.append(''.join(list(new_word)))

    #The score will be a normalized value of how successful the other videos are
    score = 0
    for item in related_item:
        #Find mathcing titles
        for i in range(len(titles)):
            if item.lower() == titles[i].translate(None,string.punctuation).lower():
                score += df['Success'][i]
    if len(related_item) != 0:
        score_list[j] = score/float(len(related_item))
    else:
        score_list[j] = df['Success'][j]
    
df['Related Score'] = score_list
df['Related Score'].hist()
P.show()
#Does this related score correlate with success/views?
P.plot(score_list,views,'b.')
P.xlabel('Related Video Score')
P.ylabel('Number of Views')
P.show()
#Determine the correlation
xvals = numpy.arange(0,1,0.1)
popt,pcov = scipy.optimize.curve_fit(linfunc, score_list, views, p0=(120000,500000))    
P.plot(score_list,views,'b.',xvals,linfunc(xvals,*popt),'r')
P.show()
print popt

#Collect attributes: 'Related Score', 'Date Good', 'comments' and output is 'Success'
data_final = {'Related Score':df['Related Score'], 'Date Good':df['Date Good'], 'Comments':df['comments'], 'Success':df['Success']}
new_df = pd.DataFrame(data_final)

#Segment data
size_train = 100
train_list = numpy.sort(random.sample(numpy.arange(0,tot_num),size_train))
train_df = new_df.drop(train_list)
test_df = new_df[ new_df.index.isin(train_list) ]
train_data = train_df.values
test_data = test_df.values

print 'Training...'
forest = RandomForestClassifier(n_estimators=50)
forest = forest.fit( train_data[0::,0:3], train_data[0::,3] )

print 'Predicting...'
output = forest.predict(test_data[0::,0:3])

#Compare the output of the model to the true output
counter = 0
for i in range(size_train):
    if output[i] == test_data[i,3]:
        counter += 1
print float(counter)/size_train          