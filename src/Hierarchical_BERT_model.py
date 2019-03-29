import os
import time
import json
import pickle
import timeit
import random
import param
import shutil
import collections
import numpy as np
import tensorflow as tf

# import stctrain_bert
import stctrain_bert_trainable
import datahelper
import stctokenizer
# import nuggetdetectionBERT as ND
import nuggetdetectionBERT_Trainable as ND
import dialogqualityBERT as DQ
import dialogquality_ndfeatureBERT as DQNDF
import stcevaluation as STCE

from scipy import stats
from collections import Counter
from gensim.models import Word2Vec
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors

import logging
logging.basicConfig(level=logging.DEBUG)

doclen = param.doclen
embsize = param.embsize
max_sent = param.max_sent
NDclasses = param.NDclasses
DQclasses = param.DQclasses
sentembsize = param.sentembsize

REMOVE_STOPWORDS = False
TO_LOWER = True
TOKEN_TYPE = 'nltk'
EMB = 'stc' # glove or stc

datahelper = datahelper.DataHelper(embedding_path="../embedding/STCWiki/STCWiki_mincount0.model.bin")
stctokenizer = stctokenizer.STCTokenizer()

def get_data():
    _, _, trainND, trainDQ, train_turns, train_masks = datahelper.get_model_train_data(
        'train',
        TOKEN_TYPE, 
        REMOVE_STOPWORDS, 
        TO_LOWER,
        EMB,
        bert=False,
    )

    _, _, devND, devDQ, dev_turns, dev_masks = datahelper.get_model_train_data(
        'dev',
        TOKEN_TYPE, 
        REMOVE_STOPWORDS, 
        TO_LOWER,
        EMB,
        bert=False,
    )

    _, _,  test_turns, test_masks = datahelper.get_model_test_data(
        TOKEN_TYPE, 
        REMOVE_STOPWORDS, 
        TO_LOWER,
        EMB,
        bert=False,
    )
    
    trainX = pickle.load(open('trainX_bert_512.p', 'rb'))
    devX = pickle.load(open('devX_bert_512.p', 'rb'))
    testX = pickle.load(open('testX_bert_512.p', 'rb'))

    testIDs = datahelper.testIDs
    trainDQA = [item['A'] for item in trainDQ]
    trainDQS = [item['S'] for item in trainDQ]
    trainDQE = [item['E'] for item in trainDQ]
    devDQA = [item['A'] for item in devDQ]
    devDQS = [item['S'] for item in devDQ]
    devDQE = [item['E'] for item in devDQ]

    dataND = [trainX, trainND, train_turns, train_masks, devX, devND, dev_turns, dev_masks, testX, test_turns, test_masks]
    dataDQA = [trainX, trainDQA, train_turns, devX, devDQA, dev_turns, testX, test_turns]
    dataDQS = [trainX, trainDQS, train_turns, devX, devDQS, dev_turns, testX, test_turns]
    dataDQE = [trainX, trainDQE, train_turns, devX, devDQE, dev_turns, testX, test_turns]
    
    dataDQA_NDF = [trainX, trainDQA, train_turns, trainND, devX, devDQA, dev_turns, devND, testX, test_turns]
    dataDQE_NDF = [trainX, trainDQE, train_turns, trainND, devX, devDQE, dev_turns, devND, testX, test_turns]
    dataDQS_NDF = [trainX, trainDQS, train_turns, trainND, devX, devDQS, dev_turns, devND, testX, test_turns]

    return dataND, dataDQA, dataDQE, dataDQS, dataDQA_NDF, dataDQE_NDF, dataDQS_NDF, testX, test_turns, test_masks, testIDs

dataND, dataDQA, dataDQE, dataDQS, dataDQA_NDF, dataDQE_NDF, dataDQS_NDF, testX, test_turns, test_masks, testIDs = get_data()

es = 3
fixed_paramsND  = {
    'epoch':100, 
    'early_stopping':es, 
    'batch_size':1,
    'lr':5e-4,
    'kp':1, 
    'hiddens':1024, 
    'Fsize':[2,2], 
    'gating':False, 
    'bn':True, 
    'method':ND.CNNRNN,
} 

fixed_paramsDQ = {
    'epoch':50, 
    'early_stopping':es, 
    'batch_size':40, 
    'lr':5e-4, 
    'kp':1, 
    'hiddens':1024, 
    'Fsize':[2,2],
    'gating':True, 
    'bn':True, 
}

BEST_PATH = 'ResultPickle/'
# bestND = pickle.load(open(BEST_PATH + 'bestND.p', "rb"))
bestDQAs = pickle.load(open(BEST_PATH + 'memoryDQAs.p', "rb"))
bestDQSs = pickle.load(open(BEST_PATH + 'memoryDQSs.p', "rb"))
bestDQEs = pickle.load(open(BEST_PATH + 'memoryDQEs.p', "rb"))

e = True
for mr in [None, 'Bi-GRU', 'Bi-LSTM']:
    for fn in [[256,512,1024]]:
        for num_layers in [1,2,3]:
            testname = 'BERT_ND{}Memory_{}stackCNN_{}stackRNN'.format(mr, len(fn), num_layers)
            print(testname)
            testND, train_losses, dev_losses = stctrain_bert_trainable.start_trainND(
                *dataND, 
                **fixed_paramsND,
                Fnum=fn, num_layers=num_layers, memory_rnn_type=mr,
                evaluate=e, bert=True
            )

    #             show_train_history(testname, train_losses, dev_losses)
            datahelper.pred_to_submission(testND, bestDQAs[0], bestDQSs[0], bestDQEs[0], test_turns, testIDs, filename='{}.json'.format(testname))