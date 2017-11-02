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

#Try LSA
'''
Returns an array with all unique words in a text
'''
def find_words(sents):
    list_words = []
    for i in  range(len(sents)):
        words = nltk.word_tokenize(sents[i])
        for j in range(len(words)):
            list_words = numpy.append(list_words, words[j].lower())
    return numpy.unique(list_words)

'''
Returns the tf for a word in a sentence
'''
def tf_word(word,sent):
    words = nltk.word_tokenize(sent)
    counter = 0
    for i in range(len(words)):
        if word.lower() == words[i].lower():
            counter += 1
    if counter != 0:
        return 1 + math.log(float(counter),10)
    else:
        return 0

'''
Returns the idf for a word
'''
def idf_word(word,sents):
    counter = 0
    for i in range(len(sents)):
        words = nltk.word_tokenize(sents[i])
        for j in range(len(words)):
            if word.lower() == words[j].lower():
                counter += 1
    if counter != 0:
        return math.log(float(len(sents))/counter,10)
    else:
        return math.log(float(len(sents))/(counter+1),10)

'''
Construct the LSA matrix
'''
def build_LSA(words_list,sents):
    LSA_mat = numpy.zeros((len(words_list),len(sents)))
    #Loop through columns in matrix
    for i in range(len(sents)):
        words = nltk.word_tokenize(sents[i])
        #For each word in this sentece calculate the tf-idf
        for k in range(len(words)):
            tf = tf_word(words[k],sents[i])
            idf = idf_word(words[k],sents)
            tf_idf = tf*idf
            for j in range(len(words_list)):
                if words_list[j].lower() == words[k].lower():
                    LSA_mat[j,i] = tf_idf
                else:
                    pass
    return LSA_mat

'''
Write matrix to text file
'''
def write_mat(filename,a):
    mat = numpy.matrix(a)
    with open(filename,'w') as f:
        for line in mat:
            numpy.savetxt(f, line, fmt='%.5f')

'''
Read in matrix
'''
def read_mat(filename):
    mat = numpy.loadtxt(filename)
    return mat

'''
Reduce dimension
'''
def LSA_svd(LSA_mat,dim):
    U, sigma, VT = numpy.linalg.svd(LSA_mat)
    #Normalize
    sigma_norm = sigma/math.sqrt(numpy.dot(sigma,sigma))
    #Reduce dimension
    sigma_norm[dim:] = 0
    S = numpy.zeros((len(U),len(VT)))
    S[:len(VT),:len(VT)] = numpy.diag(sigma_norm)
    #Return new LSA
    new_LSA = numpy.dot(U, numpy.dot(S,VT))
    return new_LSA, U, S, VT

'''
Find related words
'''
def related_words(word,words_list,U,S,num):
    term_mat = numpy.dot(U,S)
    term_index = numpy.where(words_list == word.lower())[0][0]
    #Calculate the dot product with the desired row
    correlations = numpy.zeros(len(words_list))
    for i in range(len(words_list)):
        correlations[i] = numpy.dot(term_mat[term_index,:],term_mat[i,:])/(math.sqrt(numpy.dot(term_mat[term_index,:],term_mat[term_index,:]))*math.sqrt(numpy.dot(term_mat[i,:],term_mat[i,:])))
    best_corr = numpy.argsort(correlations)[::-1][:num]
    return best_corr

'''
Find all dot products and sort by value
'''
def corr_words(words_list,U,S,num):
    term_mat = numpy.dot(U,S)
    correlations = numpy.zeros(len(words_list)*len(words_list))
    for i in range(len(words_list)):
        for j in range(len(words_list)):
            correlations[i*len(words_list)+j] = numpy.dot(term_mat[j,:],term_mat[i,:])/(math.sqrt(numpy.dot(term_mat[j,:],term_mat[j,:]))*math.sqrt(numpy.dot(term_mat[i,:],term_mat[i,:])))
    best_corr = numpy.argsort(correlations)[::-1]
    for i in range(num):
        val1 = best_corr[i]/len(words_list)
        val2 = best_corr[i]%len(words_list)
        if val1 != val2:
            print words_list[val1], words_list[val2]
    return 1

'''
Look to see if sentence by sentence the correlations are large
'''
def net_sent_corr(S,VT):
    sent_mat = numpy.dot(S,VT)
    count = 0
    for i in range(len(VT)-1):
        count += numpy.dot(sent_mat[:,i],sent_mat[:,i+1])/(math.sqrt(numpy.dot(sent_mat[:,i],sent_mat[:,i]))*math.sqrt(numpy.dot(sent_mat[:,i+1],sent_mat[:,i+1])))
    return count/float(len(VT))


#Execute
if __name__ == '__main__':
    # For .read_csv, always use header=0 when you know row 0 is the header row
    df = pd.read_csv('talks.csv', header=0)

    #Find the index for the most viewed video
    views = df['views'].values
    tags = df['related_tags']
    best_index = numpy.argsort(views)[::-1][1]
    worst_index = numpy.argsort(views)[0]   

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
    
    #Run
    best_words = find_words(sents_best)
    worst_words = find_words(sents_worst)
    print len(best_words)
    print len(sents_best)
    print len(worst_words)
    print len(sents_worst)
    #build and save matrix first time
    LSA_best = build_LSA(best_words,sents_best)
    LSA_worst = build_LSA(worst_words,sents_worst)
    write_mat('LSA_worst.txt',LSA_worst)
    write_mat('LSA_best.txt',LSA_best)
    #load in matrix
    #LSA_best = read_mat('LSA_best.txt')
    #LSA_worst = read_mat('LSA_worst.txt')
    new_LSA_best, U, S, VT = LSA_svd(LSA_best,100)
    new_LSA_worst, Uw, Sw, VTw = LSA_svd(LSA_worst,100)
    corr_words_ind = related_words(best_tags_list[0],best_words,U,S,10)
    for i in range(len(corr_words_ind)):
        print best_words[corr_words_ind[i]]
    corr_words_indw = related_words(worst_tags_list[0],worst_words,Uw,Sw,10)
    for i in range(len(corr_words_indw)):
        print worst_words[corr_words_indw[i]]
    corr_words(best_words,U,S,20)
    corr_words(worst_words,Uw,Sw,20)

    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)
    #ax.set_aspect('auto')
    #plt.imshow(all_corr, interpolation='nearest', cmap=plt.cm.ocean)
    #plt.colorbar()
    #plt.show()
    
    print net_sent_corr(S,VT)
    print net_sent_corr(Sw,VTw)