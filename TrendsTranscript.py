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
import nltk
import os
import matplotlib.pyplot as plt

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('talks.csv', header=0)

#Find the index for the most viewed video
views = df['views']
tags = df['related_tags']
best_index = numpy.argmax(views)
worst_index = numpy.argmin(views)   

#Look at the transcript for this video
best_text = df['transcript'][best_index]
worst_text = df['transcript'][worst_index]
print df['title'][best_index]
print df['title'][worst_index]

#What are the themes?
best_tags = df['related_tags'][best_index]
best_tags_list = []
s = best_tags.rsplit(',')
for word in s:
    new_word = word.translate(None, string.punctuation)
    if new_word[0] == ' ':
        hold_list = list(new_word)
        hold_list.pop(0)
        new_word = ''.join(hold_list)
    best_tags_list.append(''.join(list(new_word)))
worst_tags = df['related_tags'][worst_index]
worst_tags_list = []
s = worst_tags.rsplit(',')
for word in s:
    new_word = word.translate(None, string.punctuation)
    if new_word[0] == ' ':
        hold_list = list(new_word)
        hold_list.pop(0)
        new_word = ''.join(hold_list)
    worst_tags_list.append(''.join(list(new_word)))

f = open('best_text.txt','w')
f.write(best_text)
f.close()

f = open('worst_text.txt','w')
f.write(worst_text)
f.close()

#Keep punctuation
#tokens_best = nltk.word_tokenize(best_text)

#Strip punctuation
tokens_best = nltk.word_tokenize(best_text.translate(None, string.punctuation))
text_best = nltk.Text(tokens_best)
best_tags_list.append('Creative')
print best_tags_list
sents_best = nltk.sent_tokenize(best_text)

#Keep punctuation
#tokens_best = nltk.word_tokenize(best_text)

#Strip punctuation
tokens_worst = nltk.word_tokenize(worst_text.translate(None, string.punctuation))
text_worst = nltk.Text(tokens_worst)
worst_tags_list.append('Memories')
print worst_tags_list
sents_worst = nltk.sent_tokenize(worst_text)

#Look at frequency of common words
#Stem if needed
#stemmer = nltk.stem.SnowballStemmer("english")
#for token in tokens_best:
#    stemmer.stem(token)

fdist1 = nltk.FreqDist(text_best)
#Check to see if these match tags
for i in range(len(best_tags_list)):
    for j in range(len(fdist1.most_common())):
        if best_tags_list[i].lower() == fdist1.most_common()[j][0].lower():
            print fdist1.most_common()[j]
print 'next...'
fdist2 = nltk.FreqDist(text_worst)
#Check to see if these match tags
for i in range(len(worst_tags_list)):
    for j in range(len(fdist2.most_common())):
        if worst_tags_list[i].lower() == fdist2.most_common()[j][0].lower():
            print fdist2.most_common()[j]


#Frequency of words with a certain length best and worst
tot = len(text_best)
per_best = numpy.zeros(30)
for i in range(0,30):
    long_words = [w for w in text_best if len(w)==i]
    per_best[i] = len(long_words)/float(tot)
    
tot = len(text_worst)
per_worst = numpy.zeros(30)
for i in range(0,30):
    long_words = [w for w in text_worst if len(w)==i]
    per_worst[i] = len(long_words)/float(tot)    

#Distribution of word length in best and worst videos
P.plot(numpy.arange(0,30),per_best,'b.',numpy.arange(0,30),per_worst,'g.')
P.xlabel('Length of Word')
P.ylabel('Fraction of Occurence')
P.show()

sent_tot_best = float(len(sents_best))
sent_len_best = numpy.zeros(len(sents_best))
for i in  range(len(sents_best)):
    sent_len_best[i] = len(nltk.word_tokenize(sents_best[i]))
   
sent_tot_worst = float(len(sents_worst))                 
sent_len_worst = numpy.zeros(len(sents_worst))
for i in  range(len(sents_worst)):
    sent_len_worst[i] = len(nltk.word_tokenize(sents_worst[i]))
    
freq_best = nltk.FreqDist(sent_len_best)
freq_worst = nltk.FreqDist(sent_len_worst)

xvals_best = numpy.zeros(len(zip(*freq_best.most_common(50))[0]))
yvals_best = numpy.zeros(len(zip(*freq_best.most_common(50))[0]))
x_best = zip(*freq_best.most_common(50))[0]
y_best = zip(*freq_best.most_common(50))[1]
for i in range(len(x_best)):
    xvals_best[i], yvals_best[i] = x_best[i], y_best[i]
    
xvals_worst = numpy.zeros(len(zip(*freq_worst.most_common(50))[0]))
yvals_worst = numpy.zeros(len(zip(*freq_worst.most_common(50))[0]))
x_worst = zip(*freq_worst.most_common(50))[0]
y_worst = zip(*freq_worst.most_common(50))[1]
for i in range(len(x_worst)):
    xvals_worst[i], yvals_worst[i] = x_worst[i], y_worst[i]
#Distribution of sentence length
P.plot(xvals_best,yvals_best/sent_tot_best,'b.',xvals_worst,yvals_worst/sent_tot_worst,'g.')
P.xlabel('Length of Sentence')
P.ylabel('Fraction of Occurence')
P.show()

#Count instances of laughter
count = 0
for w in text_best:
    if w == 'Laughter':
        count += 1
print count
count = 0
for w in text_worst:
    if w == 'Laughter':
        count += 1
print count

#Frequency of tags
best_tag_list = nltk.pos_tag(text_best)
tag_fd_best = nltk.FreqDist(tag for (word, tag) in best_tag_list)
best_tag_freq = tag_fd_best.most_common()
worst_tag_list = nltk.pos_tag(text_worst)
tag_fd_worst = nltk.FreqDist(tag for (word, tag) in worst_tag_list)
worst_tag_freq = tag_fd_worst.most_common()
#Need a dictionary of the tags to match the counts with the right index
tags_dict = []
for i in range(len(best_tag_freq)):
    tags_dict.append( best_tag_freq[i][0] )
for i in range(len(worst_tag_freq)):
    tags_dict.append( worst_tag_freq[i][0] )
    
#tags_dict now lists all tags in either video 
final_tags_dict = sorted(set(tags_dict))

cum_tag_data = numpy.zeros((len(final_tags_dict),3)) 
for i in range(len(final_tags_dict)):
    for j in range(len(best_tag_freq)):
        if final_tags_dict[i] == best_tag_freq[j][0]:
            cum_tag_data[i][0] = i
            cum_tag_data[i][1] = best_tag_freq[j][1]
        else:
            pass
    for j in range(len(worst_tag_freq)):
        if final_tags_dict[i] == worst_tag_freq[j][0]:
            cum_tag_data[i][2] = best_tag_freq[j][1]
        else:
            pass
#Build a bar graph
width = 0.35
plt.bar(cum_tag_data[:,0],cum_tag_data[:,1],width,color='b')
plt.bar(cum_tag_data[:,0]+width,cum_tag_data[:,2],width,color='r')
#plt.xticks(cum_tag_data[:,0],final_tags_dict)
plt.show()

#Maybe try looking at the nouns to determine what the subject matter of each talk is?
#What is the index of the nouns and proper nouns (plural and singular)
#NN ->noun
#NNP ->proper noun
nouns_best = []
verbs_best = []
adj_best = []
for item in nltk.pos_tag(text_best):
    if item[1] == 'NN':
        nouns_best.append(item[0])
    elif item[1] == 'VB':
        verbs_best.append(item[0])
    elif item[1] == 'JJ':
        adj_best.append(item[0])
    else:
        pass
freq1_best = nltk.FreqDist(nouns_best)
freq2_best = nltk.FreqDist(verbs_best)
freq3_best = nltk.FreqDist(adj_best)
print freq1_best.most_common(20)
print freq2_best.most_common(20)
print freq3_best.most_common(20)
nouns_worst = []
verbs_worst = []
adj_worst = []
for item in nltk.pos_tag(text_worst):
    if item[1] == 'NN':
        nouns_worst.append(item[0])
    elif item[1] == 'VB':
        verbs_worst.append(item[0])
    elif item[1] == 'JJ':
        adj_worst.append(item[0])
    else:
        pass
freq1_worst = nltk.FreqDist(nouns_worst)
freq2_worst = nltk.FreqDist(verbs_worst)
freq3_worst = nltk.FreqDist(adj_worst)
print freq1_worst.most_common(20)
print freq2_worst.most_common(20)
print freq3_worst.most_common(20)